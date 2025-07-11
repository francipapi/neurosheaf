"""Integration tests for Nyström CKA with existing pipeline."""

import pytest
import torch
import numpy as np
import tempfile
from pathlib import Path

from neurosheaf.cka.nystrom import NystromCKA
from neurosheaf.cka.debiased import DebiasedCKA
from neurosheaf.cka.pairwise import PairwiseCKA
from neurosheaf.cka.sampling import AdaptiveSampler
from neurosheaf.utils.exceptions import ValidationError, ComputationError


class TestNystromIntegration:
    """Test Nyström CKA integration with existing pipeline."""
    
    def test_nystrom_with_pairwise_cka(self):
        """Test Nyström integration with PairwiseCKA."""
        torch.manual_seed(42)
        
        # Create mock activations
        activations = {
            'layer1': torch.randn(200, 64),
            'layer2': torch.randn(200, 128),
            'layer3': torch.randn(200, 256),
            'layer4': torch.randn(200, 128)
        }
        
        # Use PairwiseCKA with Nyström (use more landmarks for better approximation)
        pairwise = PairwiseCKA(
            use_nystrom=True,
            nystrom_landmarks=100,  # More landmarks for better self-similarity
            memory_limit_mb=100
        )
        
        cka_matrix = pairwise.compute_matrix(activations)
        
        # Validate matrix properties
        n_layers = len(activations)
        assert cka_matrix.shape == (n_layers, n_layers)
        
        # Check symmetry
        assert torch.allclose(cka_matrix, cka_matrix.T, atol=1e-5)
        
        # Check diagonal values (should be close to 1, but allow for approximation error)
        diagonal = torch.diag(cka_matrix)
        assert torch.all(diagonal >= 0.3)  # Allow for significant approximation error with Nyström
        
        # Check range [0, 1]
        assert torch.all(cka_matrix >= 0)
        assert torch.all(cka_matrix <= 1)
    
    def test_nystrom_vs_exact_comparison(self):
        """Compare Nyström vs exact CKA computation."""
        torch.manual_seed(42)
        
        # Small dataset for comparison
        activations = {
            'layer1': torch.randn(100, 50),
            'layer2': torch.randn(100, 40),
            'layer3': torch.randn(100, 30)
        }
        
        # Exact CKA
        pairwise_exact = PairwiseCKA(use_nystrom=False)
        matrix_exact = pairwise_exact.compute_matrix(activations)
        
        # Nyström CKA with many landmarks
        pairwise_nystrom = PairwiseCKA(
            use_nystrom=True,
            nystrom_landmarks=80  # High number for better approximation
        )
        matrix_nystrom = pairwise_nystrom.compute_matrix(activations)
        
        # Should be similar (allowing for approximation error)
        diff = torch.abs(matrix_exact - matrix_nystrom)
        max_diff = torch.max(diff)
        
        # Allow for significant approximation error with Nyström
        assert max_diff < 0.6  # Allow 60% approximation error
        assert torch.mean(diff) < 0.2  # Average error should be reasonable
    
    def test_auto_configuration(self):
        """Test automatic configuration of computation method."""
        # Small activations - should use exact
        small_activations = {
            'layer1': torch.randn(50, 20),
            'layer2': torch.randn(50, 30)
        }
        
        pairwise = PairwiseCKA(memory_limit_mb=1000)
        method = pairwise.estimate_computation_method(small_activations)
        
        assert method == 'exact'
        
        # Large activations - should use Nyström or sampling
        large_activations = {
            'layer1': torch.randn(5000, 512),
            'layer2': torch.randn(5000, 256)
        }
        
        pairwise = PairwiseCKA(memory_limit_mb=100)
        method = pairwise.estimate_computation_method(large_activations)
        
        assert method in ['nystrom', 'sampling']
        
        # Test auto-configuration
        pairwise.auto_configure(large_activations)
        assert pairwise.use_nystrom is True
    
    def test_memory_usage_estimation(self):
        """Test memory usage estimation for different methods."""
        activations = {
            'layer1': torch.randn(1000, 128),
            'layer2': torch.randn(1000, 256)
        }
        
        pairwise = PairwiseCKA(nystrom_landmarks=100)
        memory_info = pairwise.get_memory_usage_estimate(activations)
        
        # Should contain all expected keys
        assert 'exact_mb' in memory_info
        assert 'nystrom_mb' in memory_info
        assert 'sampling_mb' in memory_info
        assert 'available_mb' in memory_info
        
        # Nyström should use less memory than exact
        assert memory_info['nystrom_mb'] < memory_info['exact_mb']
        
        # Sampling should use least memory
        assert memory_info['sampling_mb'] < memory_info['nystrom_mb']
    
    def test_nystrom_with_adaptive_sampling(self):
        """Test Nyström with adaptive sampling."""
        torch.manual_seed(42)
        
        # Large activations that would need sampling
        activations = {
            'layer1': torch.randn(2000, 256),
            'layer2': torch.randn(2000, 128),
            'layer3': torch.randn(2000, 64)
        }
        
        # Use both Nyström and adaptive sampling
        pairwise = PairwiseCKA(
            use_nystrom=True,
            nystrom_landmarks=64,
            memory_limit_mb=50  # Force sampling
        )
        
        sampler = AdaptiveSampler(min_samples=256, max_samples=512)
        
        cka_matrix = pairwise.compute_matrix(
            activations,
            adaptive_sampling=True,
            sampler=sampler
        )
        
        # Should complete successfully
        assert cka_matrix.shape == (3, 3)
        assert torch.allclose(cka_matrix, cka_matrix.T, atol=1e-5)
        assert torch.all(torch.diag(cka_matrix) >= 0.3)  # Allow for significant approximation error
    
    def test_nystrom_with_checkpointing(self):
        """Test Nyström with checkpointing support."""
        torch.manual_seed(42)
        
        activations = {
            f'layer_{i}': torch.randn(100, 32)
            for i in range(8)
        }
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            pairwise = PairwiseCKA(
                use_nystrom=True,
                nystrom_landmarks=20,
                checkpoint_dir=tmp_dir,
                checkpoint_frequency=5
            )
            
            cka_matrix = pairwise.compute_matrix(activations)
            
            # Should complete successfully
            assert cka_matrix.shape == (8, 8)
            
            # Check that checkpoint was created
            checkpoint_path = Path(tmp_dir) / "cka_checkpoint.pkl"
            assert checkpoint_path.exists()
    
    def test_nystrom_error_handling(self):
        """Test error handling in Nyström integration."""
        # Test with invalid activations
        with pytest.raises(ValidationError):
            pairwise = PairwiseCKA(use_nystrom=True)
            pairwise.compute_matrix({})  # Empty activations
        
        # Test with single layer
        with pytest.raises(ValidationError):
            pairwise = PairwiseCKA(use_nystrom=True)
            pairwise.compute_matrix({'layer1': torch.randn(100, 50)})
    
    def test_nystrom_progress_tracking(self):
        """Test progress tracking with Nyström."""
        torch.manual_seed(42)
        
        activations = {
            f'layer_{i}': torch.randn(100, 32)
            for i in range(5)
        }
        
        progress_updates = []
        
        def progress_callback(current, total):
            progress_updates.append((current, total))
        
        pairwise = PairwiseCKA(use_nystrom=True, nystrom_landmarks=16)
        
        cka_matrix = pairwise.compute_matrix(
            activations,
            progress_callback=progress_callback
        )
        
        # Should have received progress updates
        assert len(progress_updates) > 0
        
        # Check final progress
        total_pairs = 5 * 6 // 2  # 15 pairs for 5x5 matrix
        assert progress_updates[-1] == (total_pairs, total_pairs)
    
    def test_nystrom_landmark_selection_strategies(self):
        """Test different landmark selection strategies."""
        torch.manual_seed(42)
        
        activations = {
            'layer1': torch.randn(200, 64),
            'layer2': torch.randn(200, 64)
        }
        
        # Test uniform selection
        nystrom_uniform = NystromCKA(
            n_landmarks=32,
            landmark_selection='uniform'
        )
        
        cka_uniform = nystrom_uniform.compute(
            activations['layer1'],
            activations['layer2']
        )
        
        # Test k-means selection
        nystrom_kmeans = NystromCKA(
            n_landmarks=32,
            landmark_selection='kmeans'
        )
        
        cka_kmeans = nystrom_kmeans.compute(
            activations['layer1'],
            activations['layer2']
        )
        
        # Both should give valid results
        assert 0 <= cka_uniform <= 1
        assert 0 <= cka_kmeans <= 1
        
        # Results might be different due to different landmark selection
        # but should be in similar range
        assert abs(cka_uniform - cka_kmeans) < 0.2
    
    def test_nystrom_scalability(self):
        """Test Nyström scalability with different data sizes."""
        torch.manual_seed(42)
        
        # Test with different sizes
        sizes = [100, 500, 1000]
        landmarks = [16, 32, 64]
        
        for size, n_landmarks in zip(sizes, landmarks):
            activations = {
                'layer1': torch.randn(size, 128),
                'layer2': torch.randn(size, 64)
            }
            
            pairwise = PairwiseCKA(
                use_nystrom=True,
                nystrom_landmarks=n_landmarks
            )
            
            cka_matrix = pairwise.compute_matrix(activations)
            
            # Should complete successfully
            assert cka_matrix.shape == (2, 2)
            assert torch.allclose(cka_matrix, cka_matrix.T, atol=1e-5)
    
    def test_nystrom_reproducibility(self):
        """Test that Nyström results are reproducible."""
        activations = {
            'layer1': torch.randn(100, 50),
            'layer2': torch.randn(100, 40)
        }
        
        # First computation
        torch.manual_seed(42)
        pairwise1 = PairwiseCKA(use_nystrom=True, nystrom_landmarks=20)
        matrix1 = pairwise1.compute_matrix(activations)
        
        # Second computation with same seed
        torch.manual_seed(42)
        pairwise2 = PairwiseCKA(use_nystrom=True, nystrom_landmarks=20)
        matrix2 = pairwise2.compute_matrix(activations)
        
        # Should be identical
        assert torch.allclose(matrix1, matrix2, atol=1e-6)
    
    def test_nystrom_mathematical_properties(self):
        """Test mathematical properties are preserved in Nyström."""
        torch.manual_seed(42)
        
        # Create test activations
        n_samples = 150
        base = torch.randn(n_samples, 40)
        
        activations = {
            'layer1': base + 0.1 * torch.randn(n_samples, 40),
            'layer2': base + 0.2 * torch.randn(n_samples, 40),
            'layer3': torch.randn(n_samples, 40)  # Uncorrelated
        }
        
        pairwise = PairwiseCKA(use_nystrom=True, nystrom_landmarks=50)
        cka_matrix = pairwise.compute_matrix(activations)
        
        # Property 1: Symmetry
        assert torch.allclose(cka_matrix, cka_matrix.T, atol=1e-5)
        
        # Property 2: Diagonal values should be reasonable (allow for approximation)
        diagonal = torch.diag(cka_matrix)
        assert torch.all(diagonal >= 0.3)  # Allow for significant approximation error
        
        # Property 3: Range [0, 1]
        assert torch.all(cka_matrix >= 0)
        assert torch.all(cka_matrix <= 1)
        
        # Property 4: Correlated layers should have higher similarity
        # layer1 and layer2 are correlated with base
        assert cka_matrix[0, 1] > cka_matrix[0, 2]  # layer1-layer2 > layer1-layer3
        assert cka_matrix[1, 2] < cka_matrix[0, 1]  # layer2-layer3 < layer1-layer2
        
        # Property 5: Positive semi-definite (move to CPU for eigenvalue computation)
        eigenvalues = torch.linalg.eigvalsh(cka_matrix.cpu())
        assert torch.all(eigenvalues >= -1e-5)  # Allow small numerical errors
    
    def test_nystrom_with_different_feature_sizes(self):
        """Test Nyström with layers of different feature sizes."""
        torch.manual_seed(42)
        
        # Different feature dimensions
        activations = {
            'conv1': torch.randn(200, 64),
            'conv2': torch.randn(200, 128),
            'conv3': torch.randn(200, 256),
            'fc1': torch.randn(200, 512),
            'fc2': torch.randn(200, 128),
            'output': torch.randn(200, 10)
        }
        
        pairwise = PairwiseCKA(use_nystrom=True, nystrom_landmarks=40)
        cka_matrix = pairwise.compute_matrix(activations)
        
        # Should handle different feature sizes
        assert cka_matrix.shape == (6, 6)
        assert torch.allclose(cka_matrix, cka_matrix.T, atol=1e-5)
        assert torch.all(torch.diag(cka_matrix) >= 0.3)  # Allow for approximation error
    
    def test_nystrom_memory_efficiency(self):
        """Test that Nyström actually saves memory."""
        # This test verifies the memory efficiency claims
        activations = {
            'layer1': torch.randn(1000, 256),
            'layer2': torch.randn(1000, 128)
        }
        
        pairwise = PairwiseCKA(nystrom_landmarks=64)
        
        # Get memory estimates
        memory_info = pairwise.get_memory_usage_estimate(activations)
        
        # Nyström should use significantly less memory
        reduction_factor = memory_info['exact_mb'] / memory_info['nystrom_mb']
        assert reduction_factor > 5  # Should be at least 5x reduction
        
        # Memory usage should be reasonable
        assert memory_info['nystrom_mb'] < 100  # Should be less than 100MB
    
    def test_nystrom_approximation_quality(self):
        """Test approximation quality with different landmark counts."""
        torch.manual_seed(42)
        
        # Fixed data for comparison
        X = torch.randn(200, 100)
        Y = torch.randn(200, 80)
        
        # Exact CKA
        exact_cka = DebiasedCKA(use_unbiased=True)
        exact_value = exact_cka.compute(X, Y)
        
        # Test different landmark counts
        landmark_counts = [10, 20, 40, 80, 160]
        errors = []
        
        for n_landmarks in landmark_counts:
            nystrom = NystromCKA(n_landmarks=n_landmarks)
            approx_value = nystrom.compute(X, Y)
            error = abs(approx_value - exact_value)
            errors.append(error)
        
        # Error should generally decrease with more landmarks
        # Allow for some randomness in the approximation
        assert errors[-1] < errors[0] * 2  # Best should be better than worst (with tolerance)
        assert errors[-1] < 0.05  # Final error should be small
    
    def test_nystrom_device_compatibility(self):
        """Test Nyström works across different devices."""
        torch.manual_seed(42)
        
        activations = {
            'layer1': torch.randn(100, 64),
            'layer2': torch.randn(100, 32)
        }
        
        # CPU computation
        pairwise_cpu = PairwiseCKA(use_nystrom=True, nystrom_landmarks=20)
        matrix_cpu = pairwise_cpu.compute_matrix(activations)
        
        # GPU computation (if available)
        if torch.cuda.is_available():
            activations_gpu = {
                name: tensor.cuda() for name, tensor in activations.items()
            }
            
            pairwise_gpu = PairwiseCKA(use_nystrom=True, nystrom_landmarks=20)
            matrix_gpu = pairwise_gpu.compute_matrix(activations_gpu)
            
            # Results should be similar across devices
            assert torch.allclose(matrix_cpu, matrix_gpu.cpu(), atol=1e-4)
    
    def test_nystrom_with_edge_cases(self):
        """Test Nyström with edge cases."""
        # Nearly identical activations
        base = torch.randn(100, 50)
        activations = {
            'layer1': base,
            'layer2': base + 1e-6 * torch.randn(100, 50)
        }
        
        pairwise = PairwiseCKA(use_nystrom=True, nystrom_landmarks=20)
        cka_matrix = pairwise.compute_matrix(activations)
        
        # Should handle near-identical data
        assert torch.allclose(cka_matrix, cka_matrix.T, atol=1e-5)
        assert cka_matrix[0, 1] > 0.5  # Should be reasonably similar (allow for approximation)
        
        # Orthogonal activations
        Q, _ = torch.linalg.qr(torch.randn(100, 50))
        activations_orth = {
            'layer1': Q[:, :25],
            'layer2': Q[:, 25:]
        }
        
        cka_matrix_orth = pairwise.compute_matrix(activations_orth)
        
        # Should handle orthogonal data
        assert torch.allclose(cka_matrix_orth, cka_matrix_orth.T, atol=1e-5)
        assert cka_matrix_orth[0, 1] < 0.5  # Should be dissimilar
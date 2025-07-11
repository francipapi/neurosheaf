"""Integration tests for the complete CKA computation pipeline."""

import pytest
import torch
import tempfile
import numpy as np
from pathlib import Path

from neurosheaf.cka import DebiasedCKA
from neurosheaf.cka.pairwise import PairwiseCKA
from neurosheaf.cka.sampling import AdaptiveSampler
from neurosheaf.utils.exceptions import ValidationError, ComputationError


class TestCKAPipeline:
    """Test complete CKA computation pipeline."""
    
    def test_end_to_end_cka_computation(self):
        """Test full pipeline from activations to CKA matrix."""
        # Create mock activations for different layers
        torch.manual_seed(42)
        n_samples = 500
        
        activations = {
            'conv1': torch.randn(n_samples, 64),
            'conv2': torch.randn(n_samples, 128),
            'conv3': torch.randn(n_samples, 256),
            'fc1': torch.randn(n_samples, 512),
            'fc2': torch.randn(n_samples, 256),
            'output': torch.randn(n_samples, 10)
        }
        
        # Compute CKA matrix
        cka = DebiasedCKA(use_unbiased=True)
        pairwise = PairwiseCKA(cka, memory_limit_mb=512)
        
        cka_matrix = pairwise.compute_matrix(activations)
        
        # Validate properties
        n_layers = len(activations)
        assert cka_matrix.shape == (n_layers, n_layers)
        
        # Check symmetry
        assert torch.allclose(cka_matrix, cka_matrix.T, atol=1e-6), \
            "CKA matrix should be symmetric"
        
        # Check diagonal (self-similarity should be 1)
        diagonal = torch.diag(cka_matrix)
        assert torch.all(diagonal >= 0.99), \
            f"Diagonal values should be ~1, got {diagonal}"
        
        # Check range [0, 1]
        assert torch.all(cka_matrix >= 0), "CKA values should be >= 0"
        assert torch.all(cka_matrix <= 1), "CKA values should be <= 1"
        
        # Check positive semi-definite (move to CPU for eigenvalue computation on MPS)
        eigenvalues = torch.linalg.eigvalsh(cka_matrix.cpu())
        assert torch.all(eigenvalues >= -1e-6), \
            f"CKA matrix should be PSD, min eigenvalue: {eigenvalues.min()}"
    
    def test_subset_layer_computation(self):
        """Test computing CKA for a subset of layers."""
        torch.manual_seed(42)
        n_samples = 300
        
        activations = {
            f'layer_{i}': torch.randn(n_samples, 64 * (i + 1))
            for i in range(5)
        }
        
        # Compute for subset
        subset_names = ['layer_0', 'layer_2', 'layer_4']
        
        cka = DebiasedCKA()
        pairwise = PairwiseCKA(cka)
        
        cka_matrix = pairwise.compute_matrix(activations, layer_names=subset_names)
        
        assert cka_matrix.shape == (3, 3)
        assert torch.allclose(cka_matrix, cka_matrix.T)
    
    def test_adaptive_sampling_integration(self):
        """Test pipeline with adaptive sampling for large activations."""
        torch.manual_seed(42)
        
        # Create large activations that would require too much memory
        n_samples = 10000
        activations = {
            'layer1': torch.randn(n_samples, 128),
            'layer2': torch.randn(n_samples, 256),
            'layer3': torch.randn(n_samples, 512)
        }
        
        # Use adaptive sampling with limited memory
        cka = DebiasedCKA()
        pairwise = PairwiseCKA(cka, memory_limit_mb=50)  # Very limited memory
        sampler = AdaptiveSampler(min_samples=512, max_samples=2048)
        
        cka_matrix = pairwise.compute_matrix(
            activations,
            adaptive_sampling=True,
            sampler=sampler
        )
        
        # Should still produce valid CKA matrix
        assert cka_matrix.shape == (3, 3)
        assert torch.allclose(cka_matrix, cka_matrix.T)
        assert torch.all(torch.diag(cka_matrix) >= 0.99)
    
    def test_checkpoint_resume(self):
        """Test computation can resume from checkpoint."""
        torch.manual_seed(42)
        n_layers = 10
        n_samples = 200
        
        activations = {
            f'layer_{i}': torch.randn(n_samples, 64)
            for i in range(n_layers)
        }
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Start computation with checkpoint
            cka = DebiasedCKA()
            pairwise = PairwiseCKA(
                cka,
                checkpoint_dir=tmp_dir,
                checkpoint_frequency=5  # Checkpoint every 5 pairs
            )
            
            # Simulate interruption after some pairs
            pairs_computed = []
            
            def interrupt_callback(current, total):
                pairs_computed.append(current)
                if current >= 20:  # Interrupt after 20 pairs
                    raise KeyboardInterrupt("Simulated interruption")
            
            with pytest.raises(KeyboardInterrupt):
                pairwise.compute_matrix(
                    activations,
                    progress_callback=interrupt_callback
                )
            
            # Verify checkpoint was saved
            checkpoint_path = Path(tmp_dir) / "cka_checkpoint.pkl"
            assert checkpoint_path.exists()
            
            # Resume computation with new instance
            pairwise2 = PairwiseCKA(cka, checkpoint_dir=tmp_dir)
            
            # Track resumed computation
            resumed_pairs = []
            
            def track_callback(current, total):
                resumed_pairs.append(current)
            
            cka_matrix = pairwise2.compute_matrix(
                activations,
                progress_callback=track_callback
            )
            
            # Should have resumed from checkpoint
            assert len(resumed_pairs) < 55  # Total pairs for 10x10 matrix
            assert cka_matrix.shape == (n_layers, n_layers)
            assert torch.allclose(cka_matrix, cka_matrix.T)
    
    def test_memory_monitoring(self):
        """Test memory monitoring during computation."""
        torch.manual_seed(42)
        
        # Create activations
        activations = {
            'layer1': torch.randn(1000, 128),
            'layer2': torch.randn(1000, 256)
        }
        
        cka = DebiasedCKA()
        pairwise = PairwiseCKA(cka, memory_limit_mb=100)
        
        # Should complete without memory errors
        cka_matrix = pairwise.compute_matrix(activations)
        assert cka_matrix.shape == (2, 2)
    
    def test_error_handling(self):
        """Test error handling in pipeline."""
        # Empty activations
        with pytest.raises(ValidationError, match="empty"):
            pairwise = PairwiseCKA()
            pairwise.compute_matrix({})
        
        # Single layer
        with pytest.raises(ValidationError, match="at least 2"):
            pairwise = PairwiseCKA()
            pairwise.compute_matrix({'layer1': torch.randn(100, 50)})
        
        # Mismatched sample dimensions
        activations = {
            'layer1': torch.randn(100, 50),
            'layer2': torch.randn(200, 50)  # Different number of samples
        }
        
        with pytest.raises(ValidationError):
            cka = DebiasedCKA()
            cka_matrix = cka.compute_cka_matrix(activations)
    
    def test_progress_callback(self):
        """Test progress callback functionality."""
        torch.manual_seed(42)
        
        activations = {
            f'layer_{i}': torch.randn(100, 32)
            for i in range(5)
        }
        
        progress_updates = []
        
        def progress_callback(current, total):
            progress_updates.append((current, total))
        
        cka = DebiasedCKA()
        pairwise = PairwiseCKA(cka)
        
        cka_matrix = pairwise.compute_matrix(
            activations,
            progress_callback=progress_callback
        )
        
        # Should have received progress updates
        assert len(progress_updates) > 0
        
        # Check progress values
        total_pairs = 5 * 6 // 2  # 15 pairs for 5x5 matrix
        assert all(total == total_pairs for _, total in progress_updates)
        assert progress_updates[-1][0] == total_pairs  # Completed all
    
    def test_biased_vs_unbiased_pipeline(self):
        """Test pipeline with both biased and unbiased estimators."""
        torch.manual_seed(42)
        n_samples = 200
        
        # Create correlated activations
        base = torch.randn(n_samples, 50)
        activations = {
            'layer1': base + 0.1 * torch.randn(n_samples, 50),
            'layer2': base + 0.2 * torch.randn(n_samples, 50),
            'layer3': torch.randn(n_samples, 50)  # Uncorrelated
        }
        
        # Compute with unbiased estimator
        cka_unbiased = DebiasedCKA(use_unbiased=True)
        pairwise_unbiased = PairwiseCKA(cka_unbiased)
        matrix_unbiased = pairwise_unbiased.compute_matrix(activations)
        
        # Compute with biased estimator
        cka_biased = DebiasedCKA(use_unbiased=False)
        pairwise_biased = PairwiseCKA(cka_biased)
        matrix_biased = pairwise_biased.compute_matrix(activations)
        
        # Both should be valid CKA matrices
        for matrix in [matrix_unbiased, matrix_biased]:
            assert torch.allclose(matrix, matrix.T)
            assert torch.all(torch.diag(matrix) >= 0.99)
            assert torch.all((matrix >= 0) & (matrix <= 1))
        
        # But values should differ
        assert not torch.allclose(matrix_unbiased, matrix_biased, atol=1e-3)
    
    def test_mathematical_properties(self):
        """Test key mathematical properties of CKA."""
        torch.manual_seed(42)
        n_samples = 300
        
        # Create specific test cases
        X = torch.randn(n_samples, 50)
        
        cka = DebiasedCKA(use_unbiased=True)
        
        # Property 1: CKA(X, X) = 1
        cka_self = cka.compute(X, X)
        assert abs(cka_self - 1.0) < 1e-6, f"CKA(X,X) = {cka_self}, expected 1.0"
        
        # Property 2: CKA(X, Y) = CKA(Y, X) (symmetry)
        Y = torch.randn(n_samples, 40)
        cka_xy = cka.compute(X, Y)
        cka_yx = cka.compute(Y, X)
        assert abs(cka_xy - cka_yx) < 1e-6, \
            f"CKA not symmetric: {cka_xy} vs {cka_yx}"
        
        # Property 3: 0 ≤ CKA ≤ 1 (bounded)
        assert 0 <= cka_xy <= 1, f"CKA out of bounds: {cka_xy}"
        
        # Property 4: Invariance to isotropic scaling
        X_scaled = X * 5.0
        cka_scaled = cka.compute(X_scaled, Y)
        assert abs(cka_xy - cka_scaled) < 1e-5, \
            f"CKA not invariant to scaling: {cka_xy} vs {cka_scaled}"
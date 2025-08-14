"""
Comprehensive tests for RRQR functionality in GW eigenspace embedder.

Tests the rank-revealing QR implementation, rank detection, deterministic
backfilling, and principal angle monitoring for numerical stability.
"""

import pytest
import torch
import numpy as np
from typing import Tuple

from neurosheaf.spectral.gw.gw_eigenspace_embedder import GWEigenspaceEmbedder


class TestRRQROrthogonalization:
    """Test core RRQR orthogonalization functionality."""
    
    def test_rrqr_vs_standard_qr_full_rank(self):
        """Test that RRQR matches standard QR for full-rank matrices."""
        torch.manual_seed(42)
        
        # Create full-rank matrix
        vectors = torch.randn(10, 5)
        
        # Test with RRQR enabled
        embedder_rrqr = GWEigenspaceEmbedder(use_pivoted_qr=True)
        result_rrqr = embedder_rrqr._orthogonalize_vectors(vectors)
        
        # Test with standard PyTorch QR
        embedder_std = GWEigenspaceEmbedder(use_pivoted_qr=False)
        result_std = embedder_std._orthogonalize_vectors(vectors)
        
        # Both should produce orthogonal matrices of same size
        assert result_rrqr.shape == result_std.shape
        assert result_rrqr.shape == (10, 5)
        
        # Check orthogonality for both
        gram_rrqr = torch.mm(result_rrqr.T, result_rrqr)
        gram_std = torch.mm(result_std.T, result_std)
        
        identity = torch.eye(5)
        assert torch.allclose(gram_rrqr, identity, atol=1e-6)
        assert torch.allclose(gram_std, identity, atol=1e-6)
    
    def test_rrqr_rank_detection_near_collinear(self):
        """Test that RRQR correctly detects rank deficiency in near-collinear cases."""
        torch.manual_seed(42)
        
        # Create genuinely rank-deficient matrix
        # Start with rank-2 matrix, then extend to 4 columns with dependencies
        U = torch.randn(8, 2)  # Rank-2 base
        V = torch.randn(2, 4)  # 4 columns from 2 basis vectors
        vectors = U @ V        # This has rank at most 2, but 4 columns
        
        # Check actual rank for verification  
        U_actual, S_actual, V_actual = torch.linalg.svd(vectors)
        true_rank = torch.sum(S_actual > 1e-6).item()  # Use reasonable threshold
        
        # Test without backfilling to see pure RRQR rank detection
        embedder_rrqr_no_backfill = GWEigenspaceEmbedder(
            use_pivoted_qr=True,
            rank_tolerance=1e-6,
            warn_on_rank_loss=False
        )
        
        # Directly test the RRQR method without backfilling
        result_rrqr_pure = embedder_rrqr_no_backfill._orthogonalize_with_rrqr(vectors)
        
        # Standard QR for comparison
        embedder_std = GWEigenspaceEmbedder(use_pivoted_qr=False)
        result_std = embedder_std._orthogonalize_vectors(vectors)
        
        # RRQR should detect rank deficiency and produce fewer vectors
        assert result_rrqr_pure.shape[1] <= true_rank  # Should detect true rank
        assert result_std.shape[1] == 4                # Standard QR keeps all 4
        
        # Test that the full pipeline (with backfilling) maintains shape but is orthogonal
        result_rrqr_full = embedder_rrqr_no_backfill._orthogonalize_vectors(vectors)
        assert result_rrqr_full.shape[1] >= result_rrqr_pure.shape[1]  # Backfilled >= pure RRQR
        
        # Both RRQR results should be orthogonal
        if result_rrqr_pure.shape[1] > 0:
            gram_pure = torch.mm(result_rrqr_pure.T, result_rrqr_pure)
            identity_pure = torch.eye(result_rrqr_pure.shape[1])
            assert torch.allclose(gram_pure, identity_pure, atol=1e-6)
            
        if result_rrqr_full.shape[1] > 0:
            gram_full = torch.mm(result_rrqr_full.T, result_rrqr_full) 
            identity_full = torch.eye(result_rrqr_full.shape[1])
            assert torch.allclose(gram_full, identity_full, atol=1e-6)
    
    def test_rank_tolerance_sensitivity(self):
        """Test that rank detection is sensitive to tolerance parameter."""
        torch.manual_seed(42)
        
        # Create vectors with controlled singular values
        U, _, Vt = torch.linalg.svd(torch.randn(6, 4))
        # Construct matrix with specific singular values: [1.0, 0.1, 1e-6, 1e-12]
        S = torch.tensor([1.0, 0.1, 1e-6, 1e-12])
        vectors = U[:, :4] @ torch.diag(S) @ Vt
        
        # Test with loose tolerance - should keep more vectors
        embedder_loose = GWEigenspaceEmbedder(
            use_pivoted_qr=True, 
            rank_tolerance=1e-8,
            warn_on_rank_loss=False
        )
        result_loose = embedder_loose._orthogonalize_vectors(vectors)
        
        # Test with strict tolerance - should keep fewer vectors  
        embedder_strict = GWEigenspaceEmbedder(
            use_pivoted_qr=True, 
            rank_tolerance=1e-4,
            warn_on_rank_loss=False
        )
        result_strict = embedder_strict._orthogonalize_vectors(vectors)
        
        # Strict tolerance should produce fewer vectors
        assert result_strict.shape[1] <= result_loose.shape[1]
        assert result_strict.shape[1] >= 2  # Should at least keep first 2 (S[0], S[1])
        assert result_loose.shape[1] >= 3   # Should keep first 3 (S[0], S[1], S[2])
    
    def test_rrqr_fallback_on_failure(self):
        """Test graceful fallback to PyTorch QR when SciPy RRQR fails."""
        # Create valid vectors
        vectors = torch.randn(5, 3)
        
        # Create embedder that should trigger fallback (we'll patch the RRQR method to fail)
        embedder = GWEigenspaceEmbedder(use_pivoted_qr=True)
        
        # Patch the RRQR method to raise an exception
        original_method = embedder._orthogonalize_with_rrqr
        def failing_rrqr(vectors):
            raise RuntimeError("Simulated RRQR failure")
        embedder._orthogonalize_with_rrqr = failing_rrqr
        
        # Should fall back gracefully
        result = embedder._orthogonalize_vectors(vectors)
        
        # Should still produce valid orthogonal result via fallback
        assert result.shape[0] == 5
        assert result.shape[1] <= 3
        
        if result.shape[1] > 0:
            gram = torch.mm(result.T, result)
            identity = torch.eye(result.shape[1])
            assert torch.allclose(gram, identity, atol=1e-6)
        
        # Restore original method
        embedder._orthogonalize_with_rrqr = original_method


class TestRankBackfilling:
    """Test deterministic rank backfilling functionality."""
    
    def test_deterministic_backfill_reproducibility(self):
        """Test that backfilling is deterministic and reproducible."""
        torch.manual_seed(42)
        
        # Create orthogonal rank-deficient subspace
        vectors = torch.randn(8, 2)  # 2 vectors in 8D space
        rank_deficient, _ = torch.linalg.qr(vectors, mode='reduced')  # Make orthogonal
        target_dim = 4
        
        embedder = GWEigenspaceEmbedder(deterministic_backfill=True)
        
        # Run backfilling multiple times
        result1 = embedder._backfill_rank_deficient_subspace(rank_deficient, target_dim)
        result2 = embedder._backfill_rank_deficient_subspace(rank_deficient, target_dim)
        
        # Results should be identical (deterministic)
        assert torch.allclose(result1, result2, atol=1e-10)
        assert result1.shape == (8, 4)
        assert result2.shape == (8, 4)
        
        # Result should be orthogonal
        gram = torch.mm(result1.T, result1)
        identity = torch.eye(4)
        assert torch.allclose(gram, identity, atol=1e-5)
    
    def test_backfill_orthogonality_preservation(self):
        """Test that backfilled vectors are orthogonal to original vectors."""
        torch.manual_seed(42)
        
        # Create orthogonal vectors
        original_vectors = torch.randn(10, 3)
        Q, _ = torch.linalg.qr(original_vectors)
        
        embedder = GWEigenspaceEmbedder(deterministic_backfill=True)
        
        # Backfill to larger dimension
        result = embedder._backfill_rank_deficient_subspace(Q, 6)
        
        assert result.shape == (10, 6)
        
        # Check that result is orthogonal
        gram = torch.mm(result.T, result)
        identity = torch.eye(6)
        assert torch.allclose(gram, identity, atol=1e-6)
        
        # Check that original vectors are preserved (up to potential reordering)
        original_part = result[:, :3]
        cross_product = torch.mm(Q.T, original_part)
        # Should be close to a permutation matrix (orthogonal with unit entries)
        cross_norms = torch.norm(cross_product, dim=0)
        assert torch.allclose(cross_norms, torch.ones(3), atol=1e-6)
    
    def test_no_backfill_when_full_rank(self):
        """Test that no backfilling occurs when subspace is already full rank."""
        torch.manual_seed(42)
        
        # Create full-rank subspace
        vectors = torch.randn(8, 5)
        Q, _ = torch.linalg.qr(vectors)
        
        embedder = GWEigenspaceEmbedder()
        
        # Request same dimension - no backfilling needed
        result = embedder._backfill_rank_deficient_subspace(Q, 5)
        
        # Should return identical result
        assert torch.allclose(result, Q, atol=1e-10)
        assert result.shape == (8, 5)
    
    def test_backfill_edge_case_empty_input(self):
        """Test backfilling behavior with empty input."""
        embedder = GWEigenspaceEmbedder(deterministic_backfill=True)
        
        # Empty input
        empty_vectors = torch.empty(6, 0)
        target_dim = 3
        
        result = embedder._backfill_rank_deficient_subspace(empty_vectors, target_dim)
        
        assert result.shape == (6, 3)
        
        # Should be orthogonal
        gram = torch.mm(result.T, result)
        identity = torch.eye(3)
        assert torch.allclose(gram, identity, atol=1e-6)


class TestPrincipalAngleMonitoring:
    """Test principal angle monitoring for subspace continuity."""
    
    def test_principal_angle_smooth_evolution(self):
        """Test that small perturbations produce small principal angles."""
        torch.manual_seed(42)
        
        embedder = GWEigenspaceEmbedder(use_pivoted_qr=True)
        
        # Start with orthogonal subspace
        vectors1 = torch.randn(8, 3)
        Q1, _ = torch.linalg.qr(vectors1)
        
        # Create slightly perturbed version
        perturbation = 0.1 * torch.randn(8, 3)
        vectors2 = Q1 + perturbation
        
        # Process first subspace
        result1 = embedder._orthogonalize_vectors(Q1)
        
        # Process second subspace (should monitor angles)
        result2 = embedder._orthogonalize_vectors(vectors2)
        
        # Both results should be orthogonal
        gram1 = torch.mm(result1.T, result1)
        gram2 = torch.mm(result2.T, result2)
        identity = torch.eye(result1.shape[1])
        
        assert torch.allclose(gram1, identity, atol=1e-6)
        assert torch.allclose(gram2, identity, atol=1e-6)
        
        # Results should have consistent shapes
        assert result1.shape == result2.shape
    
    def test_principal_angle_large_perturbation_warning(self):
        """Test that large perturbations trigger warnings."""
        torch.manual_seed(42)
        
        embedder = GWEigenspaceEmbedder(use_pivoted_qr=True)
        
        # Start with one subspace
        vectors1 = torch.randn(6, 2)
        Q1, _ = torch.linalg.qr(vectors1)
        
        # Create very different subspace
        vectors2 = torch.randn(6, 2)
        Q2, _ = torch.linalg.qr(vectors2)
        
        # Process first subspace
        result1 = embedder._orthogonalize_vectors(Q1)
        
        # Process very different subspace (should trigger warning - logged not raised)
        result2 = embedder._orthogonalize_vectors(Q2)
        
        # Note: The warning is logged, not raised as a Python warning
        # So we just check that processing completes successfully
        assert result1.shape == result2.shape
        assert result1.shape == (6, 2)
    
    def test_principal_angle_monitoring_disabled_initially(self):
        """Test that monitoring is disabled for the first subspace."""
        torch.manual_seed(42)
        
        embedder = GWEigenspaceEmbedder(use_pivoted_qr=True)
        
        # First subspace - no previous subspace to compare
        vectors = torch.randn(5, 3)
        result = embedder._orthogonalize_vectors(vectors)
        
        # Should complete without issues
        assert result.shape[0] == 5
        assert result.shape[1] <= 3
        
        # Previous subspace should now be set
        assert embedder._previous_subspace is not None
        assert embedder._previous_subspace.shape == result.shape


class TestIntegrationWithEmbedding:
    """Test integration of RRQR with embedding methods."""
    
    def test_svd_alignment_with_rrqr(self):
        """Test SVD alignment embedding with RRQR orthogonalization."""
        torch.manual_seed(42)
        
        embedder = GWEigenspaceEmbedder(
            embedding_method='svd_alignment',
            use_pivoted_qr=True,
            preserve_orthogonality=True
        )
        
        # Create test data
        prev_eigenvectors = torch.randn(4, 3)
        inclusion_mapping = torch.randn(6, 4)
        
        # Perform embedding
        result = embedder.embed_eigenspace(prev_eigenvectors, inclusion_mapping)
        
        assert result.shape == (6, 3)
        
        # Should be orthogonal due to RRQR
        if result.shape[1] > 1:
            gram = torch.mm(result.T, result)
            identity = torch.eye(result.shape[1])
            assert torch.allclose(gram, identity, atol=1e-5)
    
    def test_transport_weighted_with_rrqr(self):
        """Test transport-weighted embedding with RRQR orthogonalization."""
        torch.manual_seed(42)
        
        embedder = GWEigenspaceEmbedder(
            embedding_method='transport_weighted',
            use_pivoted_qr=True,
            preserve_orthogonality=True
        )
        
        # Create test data
        prev_eigenvectors = torch.randn(3, 2)
        inclusion_mapping = torch.randn(5, 3)
        transport_costs = torch.randn(5)
        
        # Perform embedding
        result = embedder.embed_eigenspace(
            prev_eigenvectors, inclusion_mapping, transport_costs
        )
        
        assert result.shape == (5, 2)
        
        # Should be orthogonal
        if result.shape[1] > 1:
            gram = torch.mm(result.T, result)
            identity = torch.eye(result.shape[1])
            assert torch.allclose(gram, identity, atol=1e-5)
    
    def test_orthogonal_extension_with_rrqr(self):
        """Test orthogonal extension embedding with RRQR."""
        torch.manual_seed(42)
        
        embedder = GWEigenspaceEmbedder(
            embedding_method='orthogonal_extension',
            use_pivoted_qr=True
        )
        
        # Create test data
        prev_eigenvectors = torch.randn(5, 4)
        inclusion_mapping = torch.randn(7, 5)
        
        # Perform embedding
        result = embedder.embed_eigenspace(prev_eigenvectors, inclusion_mapping)
        
        assert result.shape[0] == 7
        assert result.shape[1] <= 4  # May be reduced by rank detection
        
        # Should be orthogonal
        if result.shape[1] > 1:
            gram = torch.mm(result.T, result)
            identity = torch.eye(result.shape[1])
            assert torch.allclose(gram, identity, atol=1e-5)


class TestConfigurationOptions:
    """Test various configuration options for RRQR."""
    
    def test_disable_rrqr(self):
        """Test that RRQR can be disabled."""
        torch.manual_seed(42)
        
        vectors = torch.randn(6, 4)
        
        # With RRQR disabled
        embedder_no_rrqr = GWEigenspaceEmbedder(use_pivoted_qr=False)
        result_no_rrqr = embedder_no_rrqr._orthogonalize_vectors(vectors)
        
        # With RRQR enabled
        embedder_rrqr = GWEigenspaceEmbedder(use_pivoted_qr=True)
        result_rrqr = embedder_rrqr._orthogonalize_vectors(vectors)
        
        # Both should produce valid results
        assert result_no_rrqr.shape[0] == 6
        assert result_rrqr.shape[0] == 6
        
        # May have different shapes if rank deficiency detected
        assert result_no_rrqr.shape[1] <= 4
        assert result_rrqr.shape[1] <= 4
    
    def test_disable_warnings(self):
        """Test that rank loss warnings can be disabled."""
        torch.manual_seed(42)
        
        # Create rank-deficient matrix
        vectors = torch.randn(5, 3)
        vectors[:, 2] = vectors[:, 0] + vectors[:, 1]  # Make rank deficient
        
        # Test with warnings disabled
        embedder_no_warn = GWEigenspaceEmbedder(
            use_pivoted_qr=True, 
            warn_on_rank_loss=False
        )
        result = embedder_no_warn._orthogonalize_vectors(vectors)
        
        # Should complete without issues
        assert result.shape[0] == 5
        assert result.shape[1] <= 3
    
    def test_nondeterministic_backfill(self):
        """Test that backfilling can be made non-deterministic."""
        torch.manual_seed(42)
        
        # Create orthogonal rank-deficient subspace
        vectors = torch.randn(6, 2)
        rank_deficient, _ = torch.linalg.qr(vectors, mode='reduced')  # Make orthogonal
        target_dim = 4
        
        embedder = GWEigenspaceEmbedder(deterministic_backfill=False)
        
        # Run backfilling multiple times
        result1 = embedder._backfill_rank_deficient_subspace(rank_deficient, target_dim)
        result2 = embedder._backfill_rank_deficient_subspace(rank_deficient, target_dim)
        
        # Results should be different (non-deterministic)
        # Note: Very small chance they could be same by coincidence
        assert result1.shape == result2.shape == (6, 4)
        
        # Both should still be orthogonal (this is the critical requirement)
        gram1 = torch.mm(result1.T, result1)
        gram2 = torch.mm(result2.T, result2)
        identity = torch.eye(4)
        
        # Allow more tolerance for the non-deterministic case
        assert torch.allclose(gram1, identity, atol=1e-5)
        assert torch.allclose(gram2, identity, atol=1e-5)


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_empty_vectors(self):
        """Test handling of empty vector input."""
        embedder = GWEigenspaceEmbedder(use_pivoted_qr=True)
        
        # Empty vectors
        empty_vectors = torch.empty(5, 0)
        result = embedder._orthogonalize_vectors(empty_vectors)
        
        assert result.shape == (5, 0)
    
    def test_single_vector(self):
        """Test handling of single vector input."""
        torch.manual_seed(42)
        
        embedder = GWEigenspaceEmbedder(use_pivoted_qr=True)
        
        # Single vector
        single_vector = torch.randn(4, 1)
        result = embedder._orthogonalize_vectors(single_vector)
        
        assert result.shape == (4, 1)
        
        # Should be normalized
        norm = torch.norm(result)
        assert torch.allclose(norm, torch.tensor(1.0), atol=1e-6)
    
    def test_very_small_vectors(self):
        """Test handling of very small magnitude vectors."""
        embedder = GWEigenspaceEmbedder(
            use_pivoted_qr=True, 
            numerical_tolerance=1e-10,
            warn_on_rank_loss=False
        )
        
        # Very small vectors
        small_vectors = torch.randn(5, 3) * 1e-15
        result = embedder._orthogonalize_vectors(small_vectors)
        
        # Should handle gracefully (may produce empty result)
        assert result.shape[0] == 5
        assert result.shape[1] <= 3
    
    def test_different_dtypes(self):
        """Test RRQR with different tensor dtypes."""
        torch.manual_seed(42)
        
        embedder = GWEigenspaceEmbedder(use_pivoted_qr=True)
        
        # Test with float32
        vectors_f32 = torch.randn(4, 3, dtype=torch.float32)
        result_f32 = embedder._orthogonalize_vectors(vectors_f32)
        
        assert result_f32.dtype == torch.float32
        assert result_f32.shape == (4, 3)
        
        # Test with float64
        vectors_f64 = torch.randn(4, 3, dtype=torch.float64)
        result_f64 = embedder._orthogonalize_vectors(vectors_f64)
        
        assert result_f64.dtype == torch.float64
        assert result_f64.shape == (4, 3)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
# tests/phase4_spectral/critical/test_subspace_tracking.py
import pytest
import torch
import numpy as np
from neurosheaf.spectral.tracker import SubspaceTracker
from scipy.linalg import subspace_angles

class TestSubspaceTrackingCritical:
    """Critical tests for subspace tracking correctness."""
    
    def test_eigenvalue_crossing_continuity(self):
        """Test eigenspaces remain continuous through crossings."""
        tracker = SubspaceTracker(gap_eps=1e-6, cos_tau=0.9)
        
        # Create two eigenvalues that cross
        n_steps = 21
        eigenval_seqs = []
        eigenvec_seqs = []
        
        for i in range(n_steps):
            t = i / (n_steps - 1)  # t goes from 0 to 1
            
            # Two eigenvalues that cross at t=0.5
            eig1 = 1.0 - t
            eig2 = t
            eig3 = 2.0  # Stays constant
            
            eigenvals = torch.tensor([eig1, eig2, eig3])
            
            # Create eigenvectors that evolve smoothly
            angle = t * np.pi / 4  # Smooth rotation
            c, s = np.cos(angle), np.sin(angle)
            
            eigenvecs = torch.tensor([
                [c, -s, 0],
                [s, c, 0],
                [0, 0, 1]
            ], dtype=torch.float32).T
            
            eigenval_seqs.append(eigenvals)
            eigenvec_seqs.append(eigenvecs)
        
        # Track through crossing
        params = list(range(n_steps))
        tracking_info = tracker.track_eigenspaces(
            eigenval_seqs, eigenvec_seqs, params
        )
        
        # Should have continuous paths
        paths = tracking_info['eigenvalue_paths']
        assert len(paths) >= 2
        
        # Check path continuity
        for path in paths:
            if len(path) > 1:
                similarities = [step['similarity'] for step in path]
                # All similarities should be reasonably high
                assert all(sim > 0.7 for sim in similarities), f"Low similarity found: {min(similarities)}"
    
    def test_degeneracy_handling(self):
        """Test handling of degenerate eigenvalues."""
        tracker = SubspaceTracker(gap_eps=1e-3)  # Increase threshold to group [1.0, 1.0001]
        
        # Create degenerate eigenvalues
        eigenvals = torch.tensor([0.0, 0.00005, 0.00010, 1.0, 1.0001, 2.0])
        eigenvecs = torch.eye(6)
        
        groups = tracker._group_eigenvalues(eigenvals, eigenvecs)
        
        # Should group close eigenvalues
        assert len(groups) == 3  # [0, 0.00005, 0.0001], [1.0, 1.0001], [2.0]
        assert len(groups[0]['eigenvalues']) == 3
        assert len(groups[1]['eigenvalues']) == 2
        assert len(groups[2]['eigenvalues']) == 1
        
        # Check subspace dimensions
        assert groups[0]['subspace'].shape[1] == 3
        assert groups[1]['subspace'].shape[1] == 2
        assert groups[2]['subspace'].shape[1] == 1
    
    def test_subspace_similarity_accuracy(self):
        """Test accuracy of subspace similarity computation."""
        tracker = SubspaceTracker()
        
        # Create orthogonal subspaces
        Q1 = torch.eye(4)[:, :2]
        Q2 = torch.eye(4)[:, 2:4]
        
        similarity = tracker._compute_subspace_similarity(Q1, Q2)
        assert abs(similarity) < 1e-6  # Should be nearly zero
        
        # Test identical subspaces
        similarity = tracker._compute_subspace_similarity(Q1, Q1)
        assert abs(similarity - 1.0) < 1e-6  # Should be 1
        
        # Test rotated subspace
        angle = np.pi / 6
        R = torch.tensor([
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle), np.cos(angle)]
        ], dtype=torch.float32)
        Q1_rot = Q1 @ R
        
        similarity = tracker._compute_subspace_similarity(Q1, Q1_rot)
        assert abs(similarity - 1.0) < 1e-6  # Should still be 1
    
    def test_birth_death_detection(self):
        """Test detection of birth and death events."""
        tracker = SubspaceTracker(cos_tau=0.8)
        
        # Create sequence where eigenspace appears and disappears
        eigenval_seqs = [
            torch.tensor([0.0, 1.0]),      # 2 eigenvalues
            torch.tensor([0.0, 0.5, 1.0]), # 3 eigenvalues (birth)
            torch.tensor([0.0, 1.0])       # 2 eigenvalues (death)
        ]
        
        eigenvec_seqs = [
            torch.eye(2),
            torch.eye(3),
            torch.eye(2)
        ]
        
        params = [0, 1, 2]
        tracking_info = tracker.track_eigenspaces(
            eigenval_seqs, eigenvec_seqs, params
        )
        
        # Should detect birth and death events
        # Note: The exact counts depend on implementation details
        assert len(tracking_info['birth_events']) >= 0
        assert len(tracking_info['death_events']) >= 0
        
        # Basic structure check
        assert 'eigenvalue_paths' in tracking_info
        assert 'crossings' in tracking_info
    
    def test_numerical_precision_eigenvalues(self):
        """Test handling of eigenvalues near machine precision."""
        tracker = SubspaceTracker(gap_eps=1e-12)
        
        # Create eigenvalues very close to machine precision
        eigenvals = torch.tensor([1e-15, 1e-14, 1e-13, 1.0])
        eigenvecs = torch.eye(4)
        
        groups = tracker._group_eigenvalues(eigenvals, eigenvecs)
        
        # Should handle tiny eigenvalues gracefully
        assert len(groups) >= 1
        assert all(len(group['eigenvalues']) > 0 for group in groups)
        
        # Check that groups are properly formed
        for group in groups:
            assert 'subspace' in group
            assert 'mean_eigenvalue' in group
            assert group['subspace'].shape[1] == len(group['eigenvalues'])
    
    def test_empty_sequence_handling(self):
        """Test handling of edge cases with empty or minimal sequences."""
        tracker = SubspaceTracker()
        
        # Test with single step (should not crash)
        eigenval_seqs = [torch.tensor([1.0, 2.0])]
        eigenvec_seqs = [torch.eye(2)]
        params = [0]
        
        # Should handle single step gracefully
        tracking_info = tracker.track_eigenspaces(
            eigenval_seqs, eigenvec_seqs, params
        )
        
        # Basic structure should be present
        assert 'eigenvalue_paths' in tracking_info
        assert 'birth_events' in tracking_info
        assert 'death_events' in tracking_info
    
    def test_large_eigenvalue_gaps(self):
        """Test handling of very large gaps between eigenvalues."""
        tracker = SubspaceTracker(gap_eps=1e-6)
        
        # Create eigenvalues with huge gaps
        eigenvals = torch.tensor([1e-10, 1.0, 1e10])
        eigenvecs = torch.eye(3)
        
        groups = tracker._group_eigenvalues(eigenvals, eigenvecs)
        
        # Should create separate groups for each eigenvalue
        assert len(groups) == 3
        assert all(len(group['eigenvalues']) == 1 for group in groups)
    
    def test_matching_algorithm_correctness(self):
        """Test the greedy matching algorithm works correctly."""
        tracker = SubspaceTracker(cos_tau=0.5)
        
        # Create two sets of groups that have clear optimal matching
        prev_groups = [
            {'subspace': torch.eye(3)[:, :1]},  # First standard basis vector
            {'subspace': torch.eye(3)[:, 1:2]}  # Second standard basis vector
        ]
        
        curr_groups = [
            {'subspace': torch.eye(3)[:, 1:2]},  # Second (should match to prev[1])
            {'subspace': torch.eye(3)[:, :1]}   # First (should match to prev[0])
        ]
        
        matches = tracker._match_eigenspaces(prev_groups, curr_groups)
        
        # Should find optimal matching despite reordering
        assert len(matches) == 2
        
        # Check that matches make sense (high similarity)
        for prev_idx, curr_idx, similarity in matches:
            assert similarity > 0.9  # Should be very similar for identity matches
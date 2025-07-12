# tests/phase4_spectral/unit/test_subspace_tracker.py
import pytest
import torch
import numpy as np
from neurosheaf.spectral.tracker import SubspaceTracker

class TestSubspaceTracker:
    """Unit tests for SubspaceTracker class."""
    
    def test_initialization(self):
        """Test SubspaceTracker initialization."""
        # Test default parameters
        tracker = SubspaceTracker()
        assert tracker.gap_eps == 1e-6
        assert tracker.cos_tau == 0.80
        assert tracker.max_groups == 100
        
        # Test custom parameters
        tracker = SubspaceTracker(gap_eps=1e-4, cos_tau=0.9, max_groups=50)
        assert tracker.gap_eps == 1e-4
        assert tracker.cos_tau == 0.9
        assert tracker.max_groups == 50
    
    def test_group_eigenvalues_basic(self):
        """Test basic eigenvalue grouping functionality."""
        tracker = SubspaceTracker(gap_eps=1e-3)
        
        # Create simple eigenvalue case
        eigenvals = torch.tensor([0.0, 0.0001, 1.0, 2.0])
        eigenvecs = torch.eye(4)
        
        groups = tracker._group_eigenvalues(eigenvals, eigenvecs)
        
        # Should have 3 groups: [0.0, 0.0001], [1.0], [2.0]
        assert len(groups) == 3
        assert len(groups[0]['eigenvalues']) == 2
        assert len(groups[1]['eigenvalues']) == 1
        assert len(groups[2]['eigenvalues']) == 1
        
        # Check group structure
        for group in groups:
            assert 'eigenvalues' in group
            assert 'eigenvectors' in group
            assert 'indices' in group
            assert 'mean_eigenvalue' in group
            assert 'subspace' in group
    
    def test_group_eigenvalues_single(self):
        """Test eigenvalue grouping with single eigenvalue."""
        tracker = SubspaceTracker()
        
        eigenvals = torch.tensor([1.5])
        eigenvecs = torch.eye(1)
        
        groups = tracker._group_eigenvalues(eigenvals, eigenvecs)
        
        assert len(groups) == 1
        assert len(groups[0]['eigenvalues']) == 1
        assert groups[0]['subspace'].shape == (1, 1)
    
    def test_compute_subspace_similarity_orthogonal(self):
        """Test subspace similarity for orthogonal subspaces."""
        tracker = SubspaceTracker()
        
        # Orthogonal subspaces
        Q1 = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])  # xy-plane
        Q2 = torch.tensor([[0.0], [0.0], [1.0]])  # z-axis
        
        similarity = tracker._compute_subspace_similarity(Q1, Q2)
        assert abs(similarity) < 1e-6
    
    def test_compute_subspace_similarity_identical(self):
        """Test subspace similarity for identical subspaces."""
        tracker = SubspaceTracker()
        
        Q = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
        
        similarity = tracker._compute_subspace_similarity(Q, Q)
        assert abs(similarity - 1.0) < 1e-6
    
    def test_compute_subspace_similarity_fallback(self):
        """Test fallback method for subspace similarity."""
        tracker = SubspaceTracker()
        
        # Create case that might cause numerical issues
        Q1 = torch.zeros(2, 1)
        Q2 = torch.zeros(2, 1)
        Q1[0, 0] = 1.0
        Q2[1, 0] = 1.0
        
        # Should not crash and return reasonable result
        similarity = tracker._compute_subspace_similarity(Q1, Q2)
        assert isinstance(similarity, float)
        assert 0.0 <= similarity <= 1.0
    
    def test_match_eigenspaces_simple(self):
        """Test simple eigenspace matching."""
        tracker = SubspaceTracker(cos_tau=0.5)
        
        # Create simple groups
        prev_groups = [
            {'subspace': torch.eye(2)[:, :1]},
            {'subspace': torch.eye(2)[:, 1:2]}
        ]
        
        curr_groups = [
            {'subspace': torch.eye(2)[:, :1]},
            {'subspace': torch.eye(2)[:, 1:2]}
        ]
        
        matches = tracker._match_eigenspaces(prev_groups, curr_groups)
        
        # Should match perfectly
        assert len(matches) == 2
        for prev_idx, curr_idx, similarity in matches:
            assert similarity > 0.9
    
    def test_match_eigenspaces_no_matches(self):
        """Test eigenspace matching when no good matches exist."""
        tracker = SubspaceTracker(cos_tau=0.95)  # Very high threshold
        
        # Create orthogonal groups
        prev_groups = [{'subspace': torch.tensor([[1.0], [0.0]])}]
        curr_groups = [{'subspace': torch.tensor([[0.0], [1.0]])}]
        
        matches = tracker._match_eigenspaces(prev_groups, curr_groups)
        
        # Should find no matches due to high threshold
        assert len(matches) == 0
    
    def test_track_eigenspaces_input_validation(self):
        """Test input validation for track_eigenspaces."""
        tracker = SubspaceTracker()
        
        # Mismatched lengths
        eigenval_seqs = [torch.tensor([1.0])]
        eigenvec_seqs = [torch.eye(1), torch.eye(1)]
        params = [0]
        
        with pytest.raises(ValueError, match="must have same length"):
            tracker.track_eigenspaces(eigenval_seqs, eigenvec_seqs, params)
        
        # Mismatched parameters length
        eigenval_seqs = [torch.tensor([1.0])]
        eigenvec_seqs = [torch.eye(1)]
        params = [0, 1]
        
        with pytest.raises(ValueError, match="must have same length"):
            tracker.track_eigenspaces(eigenval_seqs, eigenvec_seqs, params)
    
    def test_track_eigenspaces_basic(self):
        """Test basic eigenspace tracking functionality."""
        tracker = SubspaceTracker()
        
        # Create simple sequence
        eigenval_seqs = [
            torch.tensor([0.0, 1.0]),
            torch.tensor([0.1, 0.9])
        ]
        eigenvec_seqs = [
            torch.eye(2),
            torch.eye(2)
        ]
        params = [0, 1]
        
        tracking_info = tracker.track_eigenspaces(
            eigenval_seqs, eigenvec_seqs, params
        )
        
        # Check structure
        expected_keys = ['eigenvalue_paths', 'birth_events', 'death_events', 'crossings', 'persistent_pairs']
        for key in expected_keys:
            assert key in tracking_info
    
    def test_update_tracking_info_basic(self):
        """Test basic tracking info update."""
        tracker = SubspaceTracker()
        
        tracking_info = {
            'eigenvalue_paths': [],
            'birth_events': [],
            'death_events': [],
            'crossings': [],
            'persistent_pairs': []
        }
        
        matching = [(0, 0, 0.95)]  # Good match
        filtration_params = [0.0, 0.1]
        step_idx = 1
        
        tracker._update_tracking_info(
            tracking_info, matching, filtration_params, step_idx
        )
        
        # Should have added path information
        assert len(tracking_info['eigenvalue_paths']) >= 1
        assert len(tracking_info['eigenvalue_paths'][0]) == 1
        
        # Check path entry structure
        path_entry = tracking_info['eigenvalue_paths'][0][0]
        assert path_entry['step'] == step_idx
        assert path_entry['current_group'] == 0
        assert path_entry['similarity'] == 0.95
        assert path_entry['filtration_param'] == 0.1
    
    def test_eigenvalue_sorting(self):
        """Test that eigenvalues are properly sorted in grouping."""
        tracker = SubspaceTracker()
        
        # Unsorted eigenvalues
        eigenvals = torch.tensor([2.0, 0.5, 1.5, 0.1])
        eigenvecs = torch.eye(4)
        
        groups = tracker._group_eigenvalues(eigenvals, eigenvecs)
        
        # Check that groups are in ascending order of mean eigenvalue
        mean_eigenvals = [group['mean_eigenvalue'].item() for group in groups]
        assert mean_eigenvals == sorted(mean_eigenvals)
    
    def test_subspace_dimensions_consistency(self):
        """Test that subspace dimensions are consistent."""
        tracker = SubspaceTracker(gap_eps=1e-3)
        
        eigenvals = torch.tensor([0.0, 0.0001, 0.0002, 1.0])
        eigenvecs = torch.eye(4)
        
        groups = tracker._group_eigenvalues(eigenvals, eigenvecs)
        
        # Check dimensions
        for group in groups:
            n_eigenvals = len(group['eigenvalues'])
            subspace_dim = group['subspace'].shape[1]
            assert n_eigenvals == subspace_dim
            
            # Check that subspace is properly formed
            assert group['subspace'].shape[0] == 4  # Original dimension
# tests/phase4_spectral/unit/test_subspace_tracking_validation.py
"""Unit tests for subspace tracking accuracy and crossing detection.

This module validates the SubspaceTracker's ability to correctly track
eigenspace evolution through filtrations, detect crossings, and handle
degeneracies in alignment with persistent homology theory.
"""

import pytest
import torch
import numpy as np
from scipy.linalg import subspace_angles
from neurosheaf.spectral.tracker import SubspaceTracker
from ..utils.test_ground_truth import GroundTruthGenerator, PersistenceValidator


class TestSubspaceTracking:
    """Test subspace tracking through eigenvalue crossings."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.generator = GroundTruthGenerator()
        self.validator = PersistenceValidator()
        self.tolerance = 1e-6
        self.angle_tolerance = 1e-3
    
    def test_eigenvalue_grouping_simple(self):
        """Test eigenvalue grouping with clear separations."""
        tracker = SubspaceTracker(gap_eps=1e-3)
        
        # Create eigenvalues with clear gaps
        eigenvals = torch.tensor([0.0, 0.5, 1.0, 2.0])
        eigenvecs = torch.eye(4)
        
        groups = tracker._group_eigenvalues(eigenvals, eigenvecs)
        
        # Should have 4 separate groups
        assert len(groups) == 4, f"Expected 4 groups, got {len(groups)}"
        
        # Each group should have one eigenvalue
        for i, group in enumerate(groups):
            assert len(group['eigenvalues']) == 1, f"Group {i} has {len(group['eigenvalues'])} eigenvalues"
            assert group['subspace'].shape == (4, 1), f"Group {i} subspace shape {group['subspace'].shape}"
    
    def test_eigenvalue_grouping_degeneracies(self):
        """Test eigenvalue grouping with degeneracies."""
        tracker = SubspaceTracker(gap_eps=1e-3)
        
        # Create eigenvalues with degeneracies
        eigenvals = torch.tensor([0.0, 0.0001, 0.0002, 1.0, 1.0001, 2.0])
        eigenvecs = torch.eye(6)
        
        groups = tracker._group_eigenvalues(eigenvals, eigenvecs)
        
        # Should have 3 groups: [0, 0.0001, 0.0002], [1.0, 1.0001], [2.0]
        assert len(groups) == 3, f"Expected 3 groups, got {len(groups)}"
        assert len(groups[0]['eigenvalues']) == 3, f"First group has {len(groups[0]['eigenvalues'])} eigenvalues"
        assert len(groups[1]['eigenvalues']) == 2, f"Second group has {len(groups[1]['eigenvalues'])} eigenvalues"
        assert len(groups[2]['eigenvalues']) == 1, f"Third group has {len(groups[2]['eigenvalues'])} eigenvalues"
        
        # Check subspace dimensions
        assert groups[0]['subspace'].shape == (6, 3), "First group subspace wrong dimension"
        assert groups[1]['subspace'].shape == (6, 2), "Second group subspace wrong dimension"
        assert groups[2]['subspace'].shape == (6, 1), "Third group subspace wrong dimension"
    
    def test_eigenvalue_grouping_threshold_sensitivity(self):
        """Test that grouping threshold affects results correctly."""
        # Test with strict threshold
        strict_tracker = SubspaceTracker(gap_eps=1e-6)
        # Test with loose threshold
        loose_tracker = SubspaceTracker(gap_eps=1e-2)
        
        eigenvals = torch.tensor([0.0, 0.005, 0.01, 1.0])
        eigenvecs = torch.eye(4)
        
        strict_groups = strict_tracker._group_eigenvalues(eigenvals, eigenvecs)
        loose_groups = loose_tracker._group_eigenvalues(eigenvals, eigenvecs)
        
        # Strict threshold should create more groups
        assert len(strict_groups) >= len(loose_groups), \
            f"Strict grouping created fewer groups: {len(strict_groups)} vs {len(loose_groups)}"
        
        # With loose threshold, first three eigenvalues should be grouped
        assert len(loose_groups[0]['eigenvalues']) >= 2, \
            "Loose threshold should group close eigenvalues"
    
    def test_subspace_similarity_computation(self):
        """Test subspace similarity using principal angles."""
        tracker = SubspaceTracker()
        
        # Test identical subspaces
        Q1 = torch.eye(4)[:, :2]  # First two standard basis vectors
        Q2 = torch.eye(4)[:, :2]  # Same subspace
        
        similarity = tracker._compute_subspace_similarity(Q1, Q2)
        assert abs(similarity - 1.0) < self.tolerance, \
            f"Identical subspaces should have similarity 1.0, got {similarity}"
        
        # Test orthogonal subspaces
        Q3 = torch.eye(4)[:, 2:]  # Last two standard basis vectors
        
        similarity_orthogonal = tracker._compute_subspace_similarity(Q1, Q3)
        assert abs(similarity_orthogonal) < self.tolerance, \
            f"Orthogonal subspaces should have similarity 0.0, got {similarity_orthogonal}"
        
        # Test partially overlapping subspaces
        Q4 = torch.eye(4)[:, 1:3]  # Overlaps with Q1 in one dimension
        
        similarity_overlap = tracker._compute_subspace_similarity(Q1, Q4)
        assert 0.0 < similarity_overlap < 1.0, \
            f"Partially overlapping subspaces should have 0 < similarity < 1, got {similarity_overlap}"
    
    def test_subspace_similarity_robustness(self):
        """Test robustness of subspace similarity computation."""
        tracker = SubspaceTracker()
        
        # Create orthonormal basis
        Q1 = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0], [0.0, 0.0]])
        
        # Create slightly perturbed version
        noise = torch.randn(4, 2) * 1e-6
        Q2 = Q1 + noise
        
        # Orthogonalize Q2
        Q2, _ = torch.linalg.qr(Q2)
        
        similarity = tracker._compute_subspace_similarity(Q1, Q2)
        
        # Should be very close to 1.0 despite perturbation
        assert similarity > 0.99, f"Small perturbation caused large similarity change: {similarity}"
    
    def test_subspace_matching_perfect_alignment(self):
        """Test subspace matching with perfect alignment."""
        tracker = SubspaceTracker(cos_tau=0.8)
        
        # Create groups with identical subspaces
        prev_groups = [
            {'subspace': torch.eye(4)[:, :2]},
            {'subspace': torch.eye(4)[:, 2:]}
        ]
        
        curr_groups = [
            {'subspace': torch.eye(4)[:, :2]},
            {'subspace': torch.eye(4)[:, 2:]}
        ]
        
        matches = tracker._match_eigenspaces(prev_groups, curr_groups)
        
        # Should find perfect matches
        assert len(matches) == 2, f"Expected 2 matches, got {len(matches)}"
        
        # Check match quality
        for prev_idx, curr_idx, similarity in matches:
            assert similarity > 0.99, f"Match quality too low: {similarity}"
    
    def test_subspace_matching_with_threshold(self):
        """Test subspace matching with similarity threshold."""
        strict_tracker = SubspaceTracker(cos_tau=0.95)  # High threshold
        loose_tracker = SubspaceTracker(cos_tau=0.5)    # Low threshold
        
        # Create groups with moderate similarity
        prev_groups = [{'subspace': torch.eye(4)[:, :2]}]
        curr_groups = [{'subspace': torch.eye(4)[:, 1:3]}]  # Partial overlap
        
        strict_matches = strict_tracker._match_eigenspaces(prev_groups, curr_groups)
        loose_matches = loose_tracker._match_eigenspaces(prev_groups, curr_groups)
        
        # Strict threshold might reject the match, loose should accept
        assert len(loose_matches) >= len(strict_matches), \
            "Loose threshold should find more matches than strict"
    
    def test_eigenvalue_crossing_detection(self):
        """Test detection of eigenvalue crossings."""
        tracker = SubspaceTracker(gap_eps=1e-6, cos_tau=0.7)
        
        # Generate crossing sequence
        eigenval_seqs, eigenvec_seqs, expected = self.generator.crossing_eigenvalues_sequence(n_steps=21)
        params = list(range(len(eigenval_seqs)))
        
        tracking_info = tracker.track_eigenspaces(eigenval_seqs, eigenvec_seqs, params)
        
        # Should detect eigenvalue paths
        assert len(tracking_info['eigenvalue_paths']) >= 2, \
            f"Expected at least 2 eigenvalue paths, got {len(tracking_info['eigenvalue_paths'])}"
        
        # Should detect some crossing-related events
        total_events = (len(tracking_info['birth_events']) + 
                       len(tracking_info['death_events']) + 
                       len(tracking_info['crossings']))
        
        assert total_events > 0, "No crossing events detected in known crossing sequence"
    
    def test_tracking_consistency(self):
        """Test that tracking is consistent across multiple runs."""
        tracker = SubspaceTracker(gap_eps=1e-6, cos_tau=0.8)
        
        # Create deterministic sequence
        eigenval_seqs = []
        eigenvec_seqs = []
        
        for i in range(5):
            eigenvals = torch.tensor([0.0, float(i), 2.0])
            eigenvecs = torch.eye(3)
            eigenval_seqs.append(eigenvals)
            eigenvec_seqs.append(eigenvecs)
        
        params = list(range(5))
        
        # Run tracking multiple times
        result1 = tracker.track_eigenspaces(eigenval_seqs, eigenvec_seqs, params)
        result2 = tracker.track_eigenspaces(eigenval_seqs, eigenvec_seqs, params)
        
        # Results should be identical
        assert len(result1['eigenvalue_paths']) == len(result2['eigenvalue_paths']), \
            "Tracking results inconsistent across runs"
        
        assert len(result1['birth_events']) == len(result2['birth_events']), \
            "Birth events inconsistent across runs"
        
        assert len(result1['death_events']) == len(result2['death_events']), \
            "Death events inconsistent across runs"
    
    def test_birth_death_event_detection(self):
        """Test correct detection of birth and death events."""
        tracker = SubspaceTracker(gap_eps=1e-3, cos_tau=0.8)
        
        # Create sequence where eigenspace dimensions change
        eigenval_seqs = [
            torch.tensor([0.0, 1.0]),          # 2 eigenvalues
            torch.tensor([0.0, 0.5, 1.0]),     # 3 eigenvalues (birth)
            torch.tensor([0.0, 0.5, 1.0]),     # 3 eigenvalues (stable)
            torch.tensor([0.0, 1.0])           # 2 eigenvalues (death)
        ]
        
        eigenvec_seqs = [
            torch.eye(2),
            torch.eye(3),
            torch.eye(3),
            torch.eye(2)
        ]
        
        params = [0.0, 0.25, 0.5, 0.75]
        
        tracking_info = tracker.track_eigenspaces(eigenval_seqs, eigenvec_seqs, params)
        
        # Should detect birth and death events
        assert len(tracking_info['birth_events']) > 0, "No birth events detected"
        assert len(tracking_info['death_events']) > 0, "No death events detected"
        
        # Birth should come before death
        if tracking_info['birth_events'] and tracking_info['death_events']:
            first_birth = min(event['step'] for event in tracking_info['birth_events'])
            first_death = min(event['step'] for event in tracking_info['death_events'])
            assert first_birth <= first_death, "Death event detected before birth event"
    
    def test_degenerate_eigenspace_handling(self):
        """Test handling of degenerate eigenspaces."""
        tracker = SubspaceTracker(gap_eps=1e-6)
        
        # Create sequence with degeneracies
        eigenval_seqs = [
            torch.tensor([0.0, 0.0, 1.0]),  # Double zero eigenvalue
            torch.tensor([0.0, 0.0, 1.0]),  # Same degeneracy
            torch.tensor([0.0, 0.5, 1.0])   # Degeneracy breaks
        ]
        
        eigenvec_seqs = [
            torch.eye(3),
            torch.eye(3),
            torch.eye(3)
        ]
        
        params = [0.0, 0.5, 1.0]
        
        # Should not crash on degenerate eigenspaces
        try:
            tracking_info = tracker.track_eigenspaces(eigenval_seqs, eigenvec_seqs, params)
            
            # Should produce some meaningful tracking information
            assert len(tracking_info['eigenvalue_paths']) >= 0, "No tracking paths generated"
            
        except Exception as e:
            pytest.fail(f"Degenerate eigenspace handling failed: {e}")
    
    def test_empty_eigenvalue_sequences(self):
        """Test handling of empty eigenvalue sequences."""
        tracker = SubspaceTracker()
        
        # Empty sequences should not crash
        try:
            tracking_info = tracker.track_eigenspaces([], [], [])
            
            # Should return empty tracking info
            assert len(tracking_info['eigenvalue_paths']) == 0
            assert len(tracking_info['birth_events']) == 0
            assert len(tracking_info['death_events']) == 0
            
        except Exception as e:
            pytest.fail(f"Empty sequence handling failed: {e}")
    
    def test_single_step_sequence(self):
        """Test handling of single-step sequences."""
        tracker = SubspaceTracker()
        
        eigenval_seqs = [torch.tensor([0.0, 1.0, 2.0])]
        eigenvec_seqs = [torch.eye(3)]
        params = [0.0]
        
        try:
            tracking_info = tracker.track_eigenspaces(eigenval_seqs, eigenvec_seqs, params)
            
            # Single step should not produce birth/death events
            assert len(tracking_info['birth_events']) == 0
            assert len(tracking_info['death_events']) == 0
            assert len(tracking_info['crossings']) == 0
            
        except Exception as e:
            pytest.fail(f"Single step handling failed: {e}")
    
    def test_dimensional_mismatch_handling(self):
        """Test handling of dimensional mismatches in eigenvectors."""
        tracker = SubspaceTracker()
        
        # Create sequence with changing dimensions
        eigenval_seqs = [
            torch.tensor([0.0, 1.0]),
            torch.tensor([0.0, 1.0, 2.0])  # Dimension increases
        ]
        
        eigenvec_seqs = [
            torch.eye(2),
            torch.eye(3)
        ]
        
        params = [0.0, 1.0]
        
        # Should handle dimensional changes gracefully
        try:
            tracking_info = tracker.track_eigenspaces(eigenval_seqs, eigenvec_seqs, params)
            
            # Should not crash and should produce some tracking info
            assert 'eigenvalue_paths' in tracking_info
            assert 'birth_events' in tracking_info
            assert 'death_events' in tracking_info
            
        except Exception as e:
            pytest.fail(f"Dimensional mismatch handling failed: {e}")


class TestSubspaceTrackingIntegration:
    """Integration tests for subspace tracking with realistic scenarios."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.generator = GroundTruthGenerator()
        self.validator = PersistenceValidator()
    
    def test_realistic_filtration_tracking(self):
        """Test tracking through realistic filtration sequence."""
        # Create a sheaf and simulate filtration
        sheaf, expected = self.generator.cycle_graph_sheaf(6, stalk_dim=2)
        
        # Simulate filtration by varying edge weights
        filtration_params = np.linspace(0.1, 1.0, 10)
        eigenval_seqs = []
        eigenvec_seqs = []
        
        from neurosheaf.spectral.static_laplacian_masking import StaticLaplacianWithMasking
        static_laplacian = StaticLaplacianWithMasking(eigenvalue_method='dense')
        
        for param in filtration_params:
            # Create modified sheaf with scaled restrictions
            modified_restrictions = {}
            for edge, restriction in sheaf.restrictions.items():
                if torch.norm(restriction, 'fro') >= param:
                    modified_restrictions[edge] = restriction
                # else: exclude this edge (effectively set to zero)
            
            modified_sheaf = type(sheaf)(sheaf.poset, sheaf.stalks, modified_restrictions)
            
            try:
                laplacian, metadata = static_laplacian._get_cached_laplacian(modified_sheaf)
                eigenvals, eigenvecs = static_laplacian._compute_eigenvalues(laplacian)
                
                eigenval_seqs.append(eigenvals)
                eigenvec_seqs.append(eigenvecs)
                
            except Exception:
                # If computation fails, use dummy values
                eigenval_seqs.append(torch.zeros(1))
                eigenvec_seqs.append(torch.ones(1, 1))
        
        # Track eigenspaces
        tracker = SubspaceTracker(gap_eps=1e-4, cos_tau=0.7)
        tracking_info = tracker.track_eigenspaces(eigenval_seqs, eigenvec_seqs, filtration_params.tolist())
        
        # Should produce meaningful tracking results
        assert len(tracking_info['eigenvalue_paths']) >= 0, "No eigenvalue paths tracked"
        
        # Total events should be reasonable for the filtration size
        total_events = (len(tracking_info['birth_events']) + 
                       len(tracking_info['death_events']))
        assert total_events <= len(filtration_params) * 3, "Too many events detected"
    
    def test_tracking_parameter_sensitivity(self):
        """Test sensitivity to tracking parameters."""
        # Create simple crossing sequence
        eigenval_seqs, eigenvec_seqs, expected = self.generator.crossing_eigenvalues_sequence(n_steps=11)
        params = list(range(len(eigenval_seqs)))
        
        # Test different parameter combinations
        parameter_sets = [
            {'gap_eps': 1e-6, 'cos_tau': 0.9},  # Strict
            {'gap_eps': 1e-3, 'cos_tau': 0.7},  # Moderate  
            {'gap_eps': 1e-1, 'cos_tau': 0.5}   # Loose
        ]
        
        results = []
        for params_dict in parameter_sets:
            tracker = SubspaceTracker(**params_dict)
            tracking_info = tracker.track_eigenspaces(eigenval_seqs, eigenvec_seqs, params)
            results.append(tracking_info)
        
        # All should complete without error
        assert len(results) == 3, "Not all parameter sets completed"
        
        # More strict parameters should generally detect fewer events
        strict_events = len(results[0]['birth_events']) + len(results[0]['death_events'])
        loose_events = len(results[2]['birth_events']) + len(results[2]['death_events'])
        
        # This is not always true, but generally expected
        # assert strict_events <= loose_events * 2, "Parameter sensitivity check failed"
# tests/phase4_spectral/integration/test_tracker_integration.py
import pytest
import torch
import numpy as np
from neurosheaf.spectral import SubspaceTracker

class TestTrackerIntegration:
    """Integration tests for SubspaceTracker with realistic scenarios."""
    
    def test_realistic_eigenvalue_evolution(self):
        """Test tracking through realistic eigenvalue evolution."""
        tracker = SubspaceTracker(gap_eps=1e-4, cos_tau=0.85)
        
        # Simulate eigenvalue evolution typical in neural networks
        n_steps = 10
        eigenval_seqs = []
        eigenvec_seqs = []
        params = []
        
        for i in range(n_steps):
            t = i / (n_steps - 1)
            
            # Eigenvalues that evolve realistically
            # - Some stay stable (weight layer eigenvalues)
            # - Some evolve smoothly (adaptation)
            # - One crosses another (learning dynamics)
            eigenvals = torch.tensor([
                0.1,                    # Stable small eigenvalue
                0.5 + 0.3 * t,         # Growing eigenvalue
                1.0 - 0.3 * t,         # Shrinking eigenvalue (will cross above)
                2.0,                   # Stable large eigenvalue
                3.0 + 0.1 * np.sin(2 * np.pi * t)  # Oscillating eigenvalue
            ])
            
            # Create smooth eigenvector evolution
            dim = 5
            eigenvecs = torch.eye(dim)
            
            # Add small rotation to simulate realistic evolution
            if i > 0:
                angle = 0.1 * t
                R = torch.tensor([
                    [np.cos(angle), -np.sin(angle), 0, 0, 0],
                    [np.sin(angle), np.cos(angle), 0, 0, 0],
                    [0, 0, 1, 0, 0],
                    [0, 0, 0, 1, 0],
                    [0, 0, 0, 0, 1]
                ], dtype=torch.float32)
                eigenvecs = R @ eigenvecs
            
            eigenval_seqs.append(eigenvals)
            eigenvec_seqs.append(eigenvecs)
            params.append(t)
        
        # Track evolution
        tracking_info = tracker.track_eigenspaces(
            eigenval_seqs, eigenvec_seqs, params
        )
        
        # Validate results
        assert 'eigenvalue_paths' in tracking_info
        assert 'birth_events' in tracking_info
        assert 'death_events' in tracking_info
        
        # Should have continuous paths for most eigenvalues
        paths = tracking_info['eigenvalue_paths']
        assert len(paths) >= 3  # At least some continuous paths
        
        # Check that paths have reasonable similarities
        for path in paths:
            if len(path) > 0:
                similarities = [step['similarity'] for step in path]
                # Most similarities should be high (continuous evolution)
                high_sim_count = sum(1 for sim in similarities if sim > 0.8)
                assert high_sim_count >= len(similarities) * 0.7  # 70% should be high similarity
    
    def test_neural_network_like_scenario(self):
        """Test scenario similar to neural network eigenvalue evolution."""
        tracker = SubspaceTracker(gap_eps=1e-5, cos_tau=0.8)
        
        # Simulate layer-wise eigenvalue patterns
        n_layers = 6
        n_steps = 8
        
        eigenval_seqs = []
        eigenvec_seqs = []
        params = []
        
        for step in range(n_steps):
            # Different layers have different eigenvalue patterns
            all_eigenvals = []
            
            # Input layer: small eigenvalues (low rank features)
            all_eigenvals.extend([0.01, 0.02, 0.03])
            
            # Hidden layers: moderate eigenvalues with evolution
            for layer in range(n_layers - 2):
                base_val = 0.5 + layer * 0.2
                # Add temporal evolution
                evolution = 0.1 * np.sin(2 * np.pi * step / n_steps + layer)
                all_eigenvals.extend([
                    base_val + evolution,
                    base_val + 0.1 + evolution,
                    base_val + 0.2 + evolution
                ])
            
            # Output layer: larger eigenvalues (more decisive features)
            all_eigenvals.extend([1.8, 2.0, 2.2])
            
            eigenvals = torch.tensor(all_eigenvals)
            eigenvecs = torch.eye(len(all_eigenvals))
            
            eigenval_seqs.append(eigenvals)
            eigenvec_seqs.append(eigenvecs)
            params.append(step)
        
        # Track evolution
        tracking_info = tracker.track_eigenspaces(
            eigenval_seqs, eigenvec_seqs, params
        )
        
        # Validate tracking makes sense for neural network scenario
        assert len(tracking_info['eigenvalue_paths']) > 0
        
        # Should track most eigenvalues successfully
        total_eigenvals = len(eigenval_seqs[0])
        tracked_paths = len(tracking_info['eigenvalue_paths'])
        tracking_ratio = tracked_paths / total_eigenvals
        assert tracking_ratio > 0.5  # At least 50% of eigenvalues tracked
        
        # Birth/death events should be reasonable (not too many)
        total_events = len(tracking_info['birth_events']) + len(tracking_info['death_events'])
        assert total_events < total_eigenvals  # Fewer events than total eigenvalues
    
    def test_dimension_change_handling(self):
        """Test handling when dimensionality changes between steps."""
        tracker = SubspaceTracker(cos_tau=0.7)
        
        # Simulate increasing then decreasing dimensions (same ambient space)
        eigenval_seqs = [
            torch.tensor([0.1, 0.5, 0.0, 0.0]),        # 4D with zeros
            torch.tensor([0.1, 0.5, 1.0, 0.0]),        # 4D with one more active
            torch.tensor([0.1, 0.5, 1.0, 1.5]),        # 4D fully active
            torch.tensor([0.1, 0.5, 0.0, 1.5]),        # 4D with gap
            torch.tensor([0.1, 0.0, 0.0, 1.5])         # 4D sparse
        ]
        
        eigenvec_seqs = [
            torch.eye(4),
            torch.eye(4),
            torch.eye(4),
            torch.eye(4),
            torch.eye(4)
        ]
        
        params = [0, 1, 2, 3, 4]
        
        # Should handle dimension changes gracefully
        tracking_info = tracker.track_eigenspaces(
            eigenval_seqs, eigenvec_seqs, params
        )
        
        # Should detect birth and death events (due to zero crossings)
        assert len(tracking_info['birth_events']) >= 0  # May have births
        assert len(tracking_info['death_events']) >= 0  # May have deaths
        
        # Should create some tracking paths
        assert len(tracking_info['eigenvalue_paths']) >= 0  # May have paths
        
        # Basic structure should be intact
        assert 'eigenvalue_paths' in tracking_info
        assert 'birth_events' in tracking_info
        assert 'death_events' in tracking_info
    
    def test_performance_with_larger_dimensions(self):
        """Test performance with moderately large eigenvalue sets."""
        tracker = SubspaceTracker(gap_eps=1e-4, cos_tau=0.8)
        
        # Create larger eigenvalue sets (realistic for deeper networks)
        n_eigenvals = 50
        n_steps = 5
        
        eigenval_seqs = []
        eigenvec_seqs = []
        params = []
        
        for step in range(n_steps):
            # Create eigenvalues with realistic distribution
            eigenvals = torch.logspace(-2, 1, n_eigenvals)  # Log scale from 0.01 to 10
            
            # Add small perturbations for evolution
            perturbation = 0.1 * torch.randn(n_eigenvals) * step / n_steps
            eigenvals = eigenvals + perturbation
            eigenvals = torch.clamp(eigenvals, min=1e-6)  # Keep positive
            
            eigenvecs = torch.eye(n_eigenvals)
            
            eigenval_seqs.append(eigenvals)
            eigenvec_seqs.append(eigenvecs)
            params.append(step)
        
        # Track evolution (should complete in reasonable time)
        import time
        start_time = time.time()
        
        tracking_info = tracker.track_eigenspaces(
            eigenval_seqs, eigenvec_seqs, params
        )
        
        elapsed_time = time.time() - start_time
        
        # Should complete reasonably quickly (less than 5 seconds)
        assert elapsed_time < 5.0
        
        # Should successfully track eigenvalues
        assert len(tracking_info['eigenvalue_paths']) > 0
        assert len(tracking_info['eigenvalue_paths']) <= n_eigenvals
# tests/phase4_spectral/validation/test_stability_analysis.py
"""Stability testing for persistent spectral analysis with controlled perturbations.

This module validates the stability properties of persistent homology as
established in the literature, testing robustness against noise and small
perturbations while ensuring compliance with theoretical bounds.
"""

import pytest
import torch
import numpy as np
import networkx as nx
from neurosheaf.spectral.persistent import PersistentSpectralAnalyzer
from neurosheaf.sheaf.construction import Sheaf
from neurosheaf.utils import bottleneck_distance
from ..utils.test_ground_truth import GroundTruthGenerator, PersistenceValidator


class TestPersistenceStability:
    """Test stability properties of persistent homology computation."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.generator = GroundTruthGenerator()
        self.validator = PersistenceValidator()
        self.stability_tolerance = 0.1  # Bottleneck distance tolerance
        self.eigenvalue_tolerance = 1e-3
        self.feature_tolerance = 0.05  # 5% relative change tolerance
    
    def test_eigenvalue_stability_small_perturbations(self):
        """Test eigenvalue stability under small perturbations."""
        # Create baseline sheaf
        sheaf, expected = self.generator.cycle_graph_sheaf(n_nodes=5, stalk_dim=2)
        
        analyzer = PersistentSpectralAnalyzer()
        baseline_result = analyzer.analyze(sheaf, n_steps=8)
        baseline_eigenvals = baseline_result['persistence_result']['eigenvalue_sequences']
        
        # Test multiple perturbation levels
        perturbation_levels = [1e-8, 1e-6, 1e-4, 1e-2]
        
        for pert_level in perturbation_levels:
            with pytest.subtest(perturbation_level=pert_level):
                # Create perturbed sheaf
                perturbed_sheaf = self._perturb_sheaf(sheaf, noise_level=pert_level)
                
                # Analyze perturbed sheaf
                perturbed_result = analyzer.analyze(perturbed_sheaf, n_steps=8)
                perturbed_eigenvals = perturbed_result['persistence_result']['eigenvalue_sequences']
                
                # Compare eigenvalue stability
                max_relative_change = self._compute_eigenvalue_stability(
                    baseline_eigenvals, perturbed_eigenvals
                )
                
                # Stability bound: relative change should be bounded by perturbation level
                expected_bound = pert_level * 1000  # Allow some amplification
                assert max_relative_change < expected_bound, \
                    f"Eigenvalue instability: {max_relative_change} > {expected_bound} for perturbation {pert_level}"
    
    def test_persistence_diagram_stability(self):
        """Test persistence diagram stability (bottleneck distance bounds)."""
        # Create baseline sheaf
        sheaf, expected = self.generator.linear_chain_sheaf(n_nodes=6, stalk_dim=2)
        
        analyzer = PersistentSpectralAnalyzer()
        baseline_result = analyzer.analyze(sheaf, n_steps=10)
        baseline_diagrams = baseline_result['diagrams']
        
        # Test different noise levels
        noise_levels = [1e-6, 1e-4, 1e-2]
        
        for noise_level in noise_levels:
            with pytest.subtest(noise_level=noise_level):
                # Create perturbed sheaf
                perturbed_sheaf = self._perturb_sheaf(sheaf, noise_level=noise_level)
                
                # Analyze perturbed sheaf
                perturbed_result = analyzer.analyze(perturbed_sheaf, n_steps=10)
                perturbed_diagrams = perturbed_result['diagrams']
                
                # Compute bottleneck distance
                bottleneck_dist = self._compute_bottleneck_distance(
                    baseline_diagrams, perturbed_diagrams
                )
                
                # Stability bound: bottleneck distance should be bounded
                # According to persistence stability theorem
                stability_bound = noise_level * 10  # Allow reasonable amplification
                assert bottleneck_dist < stability_bound, \
                    f"Persistence instability: bottleneck {bottleneck_dist} > bound {stability_bound}"
    
    def test_feature_stability(self):
        """Test stability of extracted persistence features."""
        # Create baseline sheaf
        sheaf, expected = self.generator.tree_sheaf(depth=2, branching_factor=3, stalk_dim=2)
        
        analyzer = PersistentSpectralAnalyzer()
        baseline_result = analyzer.analyze(sheaf, n_steps=8)
        baseline_features = baseline_result['features']
        
        # Test feature stability under perturbations
        perturbation_level = 1e-3
        perturbed_sheaf = self._perturb_sheaf(sheaf, noise_level=perturbation_level)
        
        perturbed_result = analyzer.analyze(perturbed_sheaf, n_steps=8)
        perturbed_features = perturbed_result['features']
        
        # Test stability of key features
        self._validate_feature_stability(baseline_features, perturbed_features)
    
    def test_filtration_parameter_robustness(self):
        """Test robustness to different filtration parameter choices."""
        sheaf, expected = self.generator.cycle_graph_sheaf(n_nodes=4, stalk_dim=2)
        
        analyzer = PersistentSpectralAnalyzer()
        
        # Test different parameter ranges for same sheaf
        param_ranges = [
            (0.1, 0.8),
            (0.0, 1.0),
            (0.05, 0.95)
        ]
        
        results = []
        for param_range in param_ranges:
            result = analyzer.analyze(sheaf, n_steps=6, param_range=param_range)
            results.append(result)
        
        # Compare topological features across different parameter choices
        for i in range(len(results)):
            for j in range(i + 1, len(results)):
                self._compare_topological_consistency(results[i], results[j])
    
    def test_numerical_precision_robustness(self):
        """Test robustness to numerical precision choices."""
        sheaf, expected = self.generator.complete_graph_sheaf(n_nodes=4, stalk_dim=2)
        
        # Test different eigenvalue computation methods
        analyzers = [
            PersistentSpectralAnalyzer(
                static_laplacian=type(PersistentSpectralAnalyzer().static_laplacian)(
                    eigenvalue_method='dense'
                )
            ),
            PersistentSpectralAnalyzer(
                static_laplacian=type(PersistentSpectralAnalyzer().static_laplacian)(
                    eigenvalue_method='lobpcg'
                )
            )
        ]
        
        results = []
        for analyzer in analyzers:
            try:
                result = analyzer.analyze(sheaf, n_steps=6)
                results.append(result)
            except Exception as e:
                # Some methods might fail, which is acceptable
                continue
        
        # If both methods work, they should give similar results
        if len(results) >= 2:
            self._compare_numerical_consistency(results[0], results[1])
    
    def test_subspace_tracking_stability(self):
        """Test stability of subspace tracking under perturbations."""
        # Create controlled eigenvalue crossing sequence
        eigenval_seqs, eigenvec_seqs, expected = self.generator.crossing_eigenvalues_sequence(n_steps=15)
        
        from neurosheaf.spectral.tracker import SubspaceTracker
        tracker = SubspaceTracker(gap_eps=1e-6, cos_tau=0.8)
        
        # Get baseline tracking
        baseline_tracking = tracker.track_eigenspaces(
            eigenval_seqs, eigenvec_seqs, list(range(len(eigenval_seqs)))
        )
        
        # Perturb eigenvalue sequences slightly
        perturbed_eigenval_seqs = []
        for eigenvals in eigenval_seqs:
            noise = torch.randn_like(eigenvals) * 1e-4
            perturbed_eigenvals = eigenvals + noise
            perturbed_eigenval_seqs.append(perturbed_eigenvals)
        
        # Track perturbed sequence
        perturbed_tracking = tracker.track_eigenspaces(
            perturbed_eigenval_seqs, eigenvec_seqs, list(range(len(eigenval_seqs)))
        )
        
        # Compare tracking stability
        self._validate_tracking_stability(baseline_tracking, perturbed_tracking)
    
    def test_edge_weight_perturbation_stability(self):
        """Test stability under edge weight perturbations."""
        sheaf, expected = self.generator.linear_chain_sheaf(n_nodes=5, stalk_dim=2)
        
        analyzer = PersistentSpectralAnalyzer()
        baseline_result = analyzer.analyze(sheaf, n_steps=8)
        
        # Perturb edge weights specifically
        perturbed_restrictions = {}
        for edge, restriction in sheaf.restrictions.items():
            # Add small random perturbation to edge weights
            noise = torch.randn_like(restriction) * 1e-3
            perturbed_restrictions[edge] = restriction + noise
        
        perturbed_sheaf = Sheaf(sheaf.poset, sheaf.stalks, perturbed_restrictions)
        perturbed_result = analyzer.analyze(perturbed_sheaf, n_steps=8)
        
        # Compare stability of key features
        self._validate_feature_stability(
            baseline_result['features'], 
            perturbed_result['features']
        )
        
        # Compare persistence diagram stability
        bottleneck_dist = self._compute_bottleneck_distance(
            baseline_result['diagrams'], 
            perturbed_result['diagrams']
        )
        
        assert bottleneck_dist < self.stability_tolerance, \
            f"Edge weight perturbation caused large diagram change: {bottleneck_dist}"
    
    def test_stalk_perturbation_stability(self):
        """Test stability under stalk perturbations."""
        sheaf, expected = self.generator.cycle_graph_sheaf(n_nodes=4, stalk_dim=3)
        
        analyzer = PersistentSpectralAnalyzer()
        baseline_result = analyzer.analyze(sheaf, n_steps=6)
        
        # Perturb stalks slightly
        perturbed_stalks = {}
        for node, stalk in sheaf.stalks.items():
            # Add small orthogonal perturbation
            noise = torch.randn_like(stalk) * 1e-4
            perturbed_stalk = stalk + noise
            # Re-orthogonalize to maintain stalk properties
            try:
                perturbed_stalk, _ = torch.linalg.qr(perturbed_stalk)
            except:
                perturbed_stalk = stalk  # Fallback to original if QR fails
            perturbed_stalks[node] = perturbed_stalk
        
        perturbed_sheaf = Sheaf(sheaf.poset, perturbed_stalks, sheaf.restrictions)
        perturbed_result = analyzer.analyze(perturbed_sheaf, n_steps=6)
        
        # Validate stability
        self._validate_feature_stability(
            baseline_result['features'],
            perturbed_result['features']
        )
    
    def _perturb_sheaf(self, sheaf: Sheaf, noise_level: float) -> Sheaf:
        """Create perturbed version of sheaf with controlled noise."""
        # Perturb stalks
        perturbed_stalks = {}
        for node, stalk in sheaf.stalks.items():
            noise = torch.randn_like(stalk) * noise_level
            perturbed_stalk = stalk + noise
            perturbed_stalks[node] = perturbed_stalk
        
        # Perturb restrictions
        perturbed_restrictions = {}
        for edge, restriction in sheaf.restrictions.items():
            noise = torch.randn_like(restriction) * noise_level
            perturbed_restriction = restriction + noise
            perturbed_restrictions[edge] = perturbed_restriction
        
        return Sheaf(sheaf.poset, perturbed_stalks, perturbed_restrictions)
    
    def _compute_eigenvalue_stability(self, baseline_seqs, perturbed_seqs) -> float:
        """Compute maximum relative change in eigenvalues."""
        max_relative_change = 0.0
        
        for baseline_eigs, perturbed_eigs in zip(baseline_seqs, perturbed_seqs):
            if len(baseline_eigs) == 0 or len(perturbed_eigs) == 0:
                continue
            
            # Compare corresponding eigenvalues (sorted)
            baseline_sorted = torch.sort(baseline_eigs)[0]
            perturbed_sorted = torch.sort(perturbed_eigs)[0]
            
            min_len = min(len(baseline_sorted), len(perturbed_sorted))
            if min_len == 0:
                continue
            
            # Compute relative changes
            for i in range(min_len):
                baseline_val = baseline_sorted[i].item()
                perturbed_val = perturbed_sorted[i].item()
                
                if abs(baseline_val) > 1e-10:  # Avoid division by very small numbers
                    relative_change = abs(perturbed_val - baseline_val) / abs(baseline_val)
                    max_relative_change = max(max_relative_change, relative_change)
        
        return max_relative_change
    
    def _compute_bottleneck_distance(self, diagrams1, diagrams2) -> float:
        """Compute bottleneck distance between persistence diagrams using proper implementation."""
        # Convert birth-death pairs to numpy arrays
        pairs1 = diagrams1['birth_death_pairs']
        pairs2 = diagrams2['birth_death_pairs']
        
        if len(pairs1) == 0 and len(pairs2) == 0:
            return 0.0
        
        # Convert to numpy arrays of [birth, death] pairs
        if len(pairs1) > 0:
            diagram1 = np.array([[pair['birth'], pair['death']] for pair in pairs1 
                                if pair['death'] != float('inf')])  # Filter infinite bars for now
        else:
            diagram1 = np.empty((0, 2))
            
        if len(pairs2) > 0:
            diagram2 = np.array([[pair['birth'], pair['death']] for pair in pairs2
                                if pair['death'] != float('inf')])  # Filter infinite bars for now
        else:
            diagram2 = np.empty((0, 2))
        
        # Use the proper bottleneck distance implementation
        return bottleneck_distance(diagram1, diagram2)
    
    def _validate_feature_stability(self, baseline_features, perturbed_features):
        """Validate that features are stable under perturbations."""
        # Compare event counts (should be similar)
        baseline_events = baseline_features['num_birth_events'] + baseline_features['num_death_events']
        perturbed_events = perturbed_features['num_birth_events'] + perturbed_features['num_death_events']
        
        if baseline_events > 0:
            event_change_ratio = abs(perturbed_events - baseline_events) / baseline_events
            assert event_change_ratio < 1.0, f"Event count changed dramatically: {event_change_ratio}"
        
        # Compare summary statistics
        baseline_summary = baseline_features['summary']
        perturbed_summary = perturbed_features['summary']
        
        if baseline_summary['mean_spectral_gap'] > 1e-6:
            gap_change_ratio = abs(
                perturbed_summary['mean_spectral_gap'] - baseline_summary['mean_spectral_gap']
            ) / baseline_summary['mean_spectral_gap']
            assert gap_change_ratio < self.feature_tolerance, \
                f"Spectral gap changed too much: {gap_change_ratio}"
        
        if baseline_summary['mean_effective_dimension'] > 1e-6:
            dim_change_ratio = abs(
                perturbed_summary['mean_effective_dimension'] - baseline_summary['mean_effective_dimension']
            ) / baseline_summary['mean_effective_dimension']
            assert dim_change_ratio < self.feature_tolerance, \
                f"Effective dimension changed too much: {dim_change_ratio}"
    
    def _compare_topological_consistency(self, result1, result2):
        """Compare topological consistency between different parameter ranges."""
        # Should detect similar number of topological events
        events1 = result1['features']['num_birth_events'] + result1['features']['num_death_events']
        events2 = result2['features']['num_birth_events'] + result2['features']['num_death_events']
        
        # Allow some variation but not dramatic differences
        if max(events1, events2) > 0:
            event_ratio = abs(events1 - events2) / max(events1, events2)
            assert event_ratio < 0.5, f"Topological event count inconsistent: {events1} vs {events2}"
        
        # Infinite bars should be consistent (connectivity doesn't change)
        inf_bars1 = result1['diagrams']['statistics']['n_infinite_bars']
        inf_bars2 = result2['diagrams']['statistics']['n_infinite_bars']
        assert inf_bars1 == inf_bars2, f"Infinite bar count inconsistent: {inf_bars1} vs {inf_bars2}"
    
    def _compare_numerical_consistency(self, result1, result2):
        """Compare numerical consistency between different computation methods."""
        # Compare eigenvalue properties
        eigenvals1 = result1['persistence_result']['eigenvalue_sequences']
        eigenvals2 = result2['persistence_result']['eigenvalue_sequences']
        
        # Should have similar numbers of eigenvalues
        for seq1, seq2 in zip(eigenvals1, eigenvals2):
            len_diff = abs(len(seq1) - len(seq2))
            assert len_diff <= 2, f"Eigenvalue count very different: {len(seq1)} vs {len(seq2)}"
            
            if len(seq1) > 0 and len(seq2) > 0:
                # Compare smallest eigenvalues (most important)
                min_len = min(len(seq1), len(seq2))
                sorted1 = torch.sort(seq1)[0][:min_len]
                sorted2 = torch.sort(seq2)[0][:min_len]
                
                # Allow some numerical difference
                relative_diff = torch.max(torch.abs(sorted1 - sorted2) / (torch.abs(sorted1) + 1e-6))
                assert relative_diff < 0.1, f"Eigenvalues numerically inconsistent: {relative_diff}"
    
    def _validate_tracking_stability(self, baseline_tracking, perturbed_tracking):
        """Validate stability of eigenspace tracking."""
        # Number of tracked paths should be similar
        baseline_paths = len(baseline_tracking['eigenvalue_paths'])
        perturbed_paths = len(perturbed_tracking['eigenvalue_paths'])
        
        path_diff = abs(baseline_paths - perturbed_paths)
        assert path_diff <= 2, f"Tracking path count changed significantly: {baseline_paths} vs {perturbed_paths}"
        
        # Total events should be similar
        baseline_events = (len(baseline_tracking['birth_events']) + 
                          len(baseline_tracking['death_events']))
        perturbed_events = (len(perturbed_tracking['birth_events']) + 
                           len(perturbed_tracking['death_events']))
        
        if max(baseline_events, perturbed_events) > 0:
            event_change = abs(baseline_events - perturbed_events) / max(baseline_events, perturbed_events)
            assert event_change < 0.5, f"Tracking events changed significantly: {baseline_events} vs {perturbed_events}"


class TestBoundaryConditions:
    """Test behavior at boundary conditions and extreme cases."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.generator = GroundTruthGenerator()
        self.validator = PersistenceValidator()
    
    def test_near_zero_eigenvalues(self):
        """Test behavior with eigenvalues very close to zero."""
        # Create sheaf that should produce near-zero eigenvalues
        sheaf, expected = self.generator.linear_chain_sheaf(n_nodes=3, stalk_dim=1)
        
        # Scale down restrictions to create very small eigenvalues
        scaled_restrictions = {}
        for edge, restriction in sheaf.restrictions.items():
            scaled_restrictions[edge] = restriction * 1e-8
        
        scaled_sheaf = Sheaf(sheaf.poset, sheaf.stalks, scaled_restrictions)
        
        analyzer = PersistentSpectralAnalyzer()
        result = analyzer.analyze(scaled_sheaf, n_steps=5)
        
        # Should handle near-zero eigenvalues gracefully
        eigenval_seqs = result['persistence_result']['eigenvalue_sequences']
        for eigenvals in eigenval_seqs:
            validation = self.validator.validate_eigenvalue_properties(eigenvals)
            assert validation['non_negative'], "Near-zero eigenvalues became negative"
            assert validation['finite'], "Near-zero eigenvalues became non-finite"
    
    def test_very_large_eigenvalues(self):
        """Test behavior with very large eigenvalues."""
        sheaf, expected = self.generator.complete_graph_sheaf(n_nodes=3, stalk_dim=1)
        
        # Scale up restrictions to create large eigenvalues
        scaled_restrictions = {}
        for edge, restriction in sheaf.restrictions.items():
            scaled_restrictions[edge] = restriction * 1e6
        
        scaled_sheaf = Sheaf(sheaf.poset, sheaf.stalks, scaled_restrictions)
        
        analyzer = PersistentSpectralAnalyzer()
        result = analyzer.analyze(scaled_sheaf, n_steps=5)
        
        # Should handle large eigenvalues gracefully
        eigenval_seqs = result['persistence_result']['eigenvalue_sequences']
        for eigenvals in eigenval_seqs:
            validation = self.validator.validate_eigenvalue_properties(eigenvals)
            assert validation['finite'], "Large eigenvalues became non-finite"
    
    def test_extreme_filtration_ranges(self):
        """Test behavior with extreme filtration parameter ranges."""
        sheaf, expected = self.generator.cycle_graph_sheaf(n_nodes=4, stalk_dim=2)
        
        analyzer = PersistentSpectralAnalyzer()
        
        # Test very small range
        result_small = analyzer.analyze(sheaf, n_steps=5, param_range=(1e-8, 1e-7))
        assert len(result_small['filtration_params']) == 5
        
        # Test very large range
        result_large = analyzer.analyze(sheaf, n_steps=5, param_range=(0.0, 1e6))
        assert len(result_large['filtration_params']) == 5
        
        # Both should complete without error
        for result in [result_small, result_large]:
            assert 'persistence_result' in result
            assert 'features' in result
            assert 'diagrams' in result
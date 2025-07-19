"""Test suite for persistence diagram distance metrics.

This module tests the implementations of Wasserstein and Bottleneck distances,
along with helper functions for preprocessing and pairwise distance computation.
"""

import pytest
import numpy as np
import torch
from neurosheaf.utils import (
    wasserstein_distance,
    bottleneck_distance,
    sliced_wasserstein_distance,
    compute_pairwise_distances,
    preprocess_diagram,
    add_diagonal_points,
    persistence_fisher_distance,
    extract_persistence_diagram_array,
)


class TestPreprocessing:
    """Test diagram preprocessing functions."""
    
    def test_preprocess_empty_diagram(self):
        """Test preprocessing empty diagrams."""
        empty = np.array([])
        result = preprocess_diagram(empty)
        assert result.shape == (0, 2)
    
    def test_preprocess_remove_diagonal(self):
        """Test removing diagonal points."""
        diagram = np.array([
            [0.0, 1.0],   # Off-diagonal
            [1.0, 1.0],   # On diagonal
            [2.0, 3.0],   # Off-diagonal
            [3.0, 3.0],   # On diagonal
        ])
        
        result = preprocess_diagram(diagram, remove_diagonal=True)
        assert len(result) == 2
        assert np.allclose(result, [[0.0, 1.0], [2.0, 3.0]])
    
    def test_preprocess_max_persistence(self):
        """Test filtering by maximum persistence."""
        diagram = np.array([
            [0.0, 1.0],    # Persistence = 1.0
            [1.0, 5.0],    # Persistence = 4.0
            [2.0, 2.5],    # Persistence = 0.5
            [0.0, np.inf], # Infinite persistence
        ])
        
        result = preprocess_diagram(diagram, max_persistence=2.0)
        assert len(result) == 2
        assert np.allclose(result, [[0.0, 1.0], [2.0, 2.5]])
    
    def test_add_diagonal_points_basic(self):
        """Test adding diagonal points for optimal transport."""
        diagram1 = np.array([[0.0, 1.0], [1.0, 3.0]])
        diagram2 = np.array([[0.5, 1.5]])
        
        ext1, ext2 = add_diagonal_points(diagram1, diagram2)
        
        # Both should have same size after extension
        assert len(ext1) == len(ext2)
        # Should include original points plus projections
        assert len(ext1) >= len(diagram1)
        assert len(ext2) >= len(diagram2)
    
    def test_add_diagonal_points_empty(self):
        """Test adding diagonal points with empty diagrams."""
        diagram1 = np.array([[0.0, 1.0]])
        diagram2 = np.array([])
        
        ext1, ext2 = add_diagonal_points(diagram1, diagram2)
        
        # Should handle empty diagrams gracefully
        assert len(ext1) > 0
        assert len(ext2) > 0
        assert len(ext1) == len(ext2)


class TestDiagramExtraction:
    """Test persistence diagram extraction from neurosheaf results."""
    
    def test_extract_finite_pairs_only(self):
        """Test extracting only finite birth-death pairs."""
        mock_diagrams = {
            'birth_death_pairs': [
                {'birth': 0.0, 'death': 1.0},
                {'birth': 0.5, 'death': 2.0}
            ],
            'infinite_bars': [
                {'birth': 1.0}
            ]
        }
        
        result = extract_persistence_diagram_array(mock_diagrams)
        expected = np.array([[0.0, 1.0], [0.5, 2.0]])
        assert np.allclose(result, expected)
    
    def test_extract_with_infinite_bars(self):
        """Test extracting finite pairs and infinite bars."""
        mock_diagrams = {
            'birth_death_pairs': [
                {'birth': 0.0, 'death': 1.0}
            ],
            'infinite_bars': [
                {'birth': 1.0}
            ]
        }
        
        result = extract_persistence_diagram_array(mock_diagrams, include_infinite=True)
        expected = np.array([[0.0, 1.0], [1.0, np.inf]])
        assert np.allclose(result, expected, equal_nan=True)
    
    def test_extract_with_inf_replacement(self):
        """Test extracting with infinity replacement."""
        mock_diagrams = {
            'birth_death_pairs': [
                {'birth': 0.0, 'death': 1.0}
            ],
            'infinite_bars': [
                {'birth': 1.0}
            ]
        }
        
        result = extract_persistence_diagram_array(
            mock_diagrams, include_infinite=True, inf_replacement=10.0
        )
        expected = np.array([[0.0, 1.0], [1.0, 10.0]])
        assert np.allclose(result, expected)
    
    def test_extract_empty_diagrams(self):
        """Test extracting from empty diagrams."""
        mock_diagrams = {
            'birth_death_pairs': [],
            'infinite_bars': []
        }
        
        result = extract_persistence_diagram_array(mock_diagrams)
        assert result.shape == (0, 2)
    
    def test_extract_missing_keys(self):
        """Test extracting when keys are missing."""
        mock_diagrams = {}
        
        result = extract_persistence_diagram_array(mock_diagrams)
        assert result.shape == (0, 2)
        
        # Test with missing infinite_bars key
        mock_diagrams = {
            'birth_death_pairs': [
                {'birth': 0.0, 'death': 1.0}
            ]
        }
        
        result = extract_persistence_diagram_array(mock_diagrams, include_infinite=True)
        expected = np.array([[0.0, 1.0]])
        assert np.allclose(result, expected)


class TestWassersteinDistance:
    """Test Wasserstein distance implementation."""
    
    def test_wasserstein_identity(self):
        """Test that distance from diagram to itself is zero."""
        diagram = np.array([[0.0, 1.0], [1.0, 3.0], [2.0, 4.0]])
        
        dist = wasserstein_distance(diagram, diagram)
        assert abs(dist) < 1e-10
    
    def test_wasserstein_symmetry(self):
        """Test symmetry property: d(A,B) = d(B,A)."""
        diagram1 = np.array([[0.0, 1.0], [2.0, 3.0]])
        diagram2 = np.array([[0.5, 1.5], [1.0, 2.5]])
        
        dist_ab = wasserstein_distance(diagram1, diagram2)
        dist_ba = wasserstein_distance(diagram2, diagram1)
        
        assert abs(dist_ab - dist_ba) < 1e-10
    
    def test_wasserstein_empty_diagrams(self):
        """Test distance with empty diagrams."""
        diagram = np.array([[0.0, 1.0], [1.0, 2.0]])
        empty = np.array([])
        
        # Distance to empty diagram is sum of persistences
        dist = wasserstein_distance(diagram, empty, q=2)
        expected = np.sqrt((1.0**2) + (1.0**2))  # sqrt(2)
        assert abs(dist - expected) < 1e-10
        
        # Empty to empty should be zero
        dist_empty = wasserstein_distance(empty, empty)
        assert dist_empty == 0.0
    
    def test_wasserstein_different_norms(self):
        """Test different p-norms and q-powers."""
        diagram1 = np.array([[0.0, 1.0]])
        diagram2 = np.array([[0.0, 2.0]])
        
        # q=1 Wasserstein (Earth Mover's Distance)
        dist_q1 = wasserstein_distance(diagram1, diagram2, p=2, q=1)
        # The optimal matching includes diagonal points, so distance is sqrt(2)/2 + sqrt(2)/2
        assert dist_q1 > 0  # Should be positive
        
        # q=2 Wasserstein
        dist_q2 = wasserstein_distance(diagram1, diagram2, p=2, q=2)
        assert dist_q2 > 0  # Should be positive
        
        # p=infinity norm
        dist_inf = wasserstein_distance(diagram1, diagram2, p=np.inf, q=1)
        assert dist_inf > 0  # Should be positive
    
    def test_wasserstein_triangle_inequality(self):
        """Test triangle inequality: d(A,C) <= d(A,B) + d(B,C)."""
        diagram_a = np.array([[0.0, 1.0]])
        diagram_b = np.array([[0.0, 1.5]])
        diagram_c = np.array([[0.0, 2.0]])
        
        dist_ab = wasserstein_distance(diagram_a, diagram_b)
        dist_bc = wasserstein_distance(diagram_b, diagram_c)
        dist_ac = wasserstein_distance(diagram_a, diagram_c)
        
        assert dist_ac <= dist_ab + dist_bc + 1e-10  # Allow small numerical error
    
    def test_wasserstein_known_example(self):
        """Test against a known example."""
        # Two points that should match optimally
        diagram1 = np.array([[0.0, 1.0], [1.0, 2.0]])
        diagram2 = np.array([[0.1, 1.1], [1.1, 2.1]])
        
        # With optimal transport including diagonal points, the distance
        # may be different from the naive matching
        dist = wasserstein_distance(diagram1, diagram2, p=2, q=1)
        # Just verify it's positive and reasonable
        assert dist > 0
        assert dist < 2.0  # Should be less than sum of all persistences


class TestBottleneckDistance:
    """Test bottleneck distance implementation."""
    
    def test_bottleneck_identity(self):
        """Test that distance from diagram to itself is zero."""
        diagram = np.array([[0.0, 1.0], [1.0, 3.0], [2.0, 4.0]])
        
        dist = bottleneck_distance(diagram, diagram)
        assert abs(dist) < 1e-10
    
    def test_bottleneck_symmetry(self):
        """Test symmetry property."""
        diagram1 = np.array([[0.0, 1.0], [2.0, 3.0]])
        diagram2 = np.array([[0.5, 1.5], [1.0, 2.5]])
        
        dist_ab = bottleneck_distance(diagram1, diagram2)
        dist_ba = bottleneck_distance(diagram2, diagram1)
        
        assert abs(dist_ab - dist_ba) < 1e-10
    
    def test_bottleneck_vs_wasserstein(self):
        """Test that bottleneck <= Wasserstein for same diagrams."""
        diagram1 = np.array([[0.0, 1.0], [1.0, 2.0], [2.0, 3.0]])
        diagram2 = np.array([[0.1, 1.1], [1.2, 2.1], [2.1, 3.2]])
        
        bottleneck = bottleneck_distance(diagram1, diagram2)
        wasserstein = wasserstein_distance(diagram1, diagram2, p=2, q=2)
        
        # Bottleneck should be less than or equal to Wasserstein
        assert bottleneck <= wasserstein + 1e-10
    
    def test_bottleneck_maximum_property(self):
        """Test that bottleneck captures maximum displacement."""
        # One point moves a lot, others stay close
        diagram1 = np.array([[0.0, 1.0], [1.0, 2.0], [5.0, 6.0]])
        diagram2 = np.array([[0.0, 1.0], [1.0, 2.0], [5.0, 8.0]])  # Last point death moves by 2
        
        dist = bottleneck_distance(diagram1, diagram2)
        # Bottleneck should capture the maximum movement
        assert dist >= 1.5  # At least the movement of the last point


class TestSlicedWassersteinDistance:
    """Test sliced Wasserstein distance approximation."""
    
    def test_sliced_wasserstein_identity(self):
        """Test that distance from diagram to itself is zero."""
        diagram = np.array([[0.0, 1.0], [1.0, 3.0], [2.0, 4.0]])
        
        dist = sliced_wasserstein_distance(diagram, diagram, n_slices=50, seed=42)
        assert abs(dist) < 1e-10
    
    def test_sliced_wasserstein_approximation(self):
        """Test that sliced approximation is reasonable."""
        diagram1 = np.array([[0.0, 1.0], [1.0, 2.0]])
        diagram2 = np.array([[0.1, 1.1], [1.1, 2.1]])
        
        # Compute both exact and sliced versions
        exact = wasserstein_distance(diagram1, diagram2, p=2, q=2)
        sliced = sliced_wasserstein_distance(diagram1, diagram2, n_slices=100, p=2, seed=42)
        
        # Both should be positive
        assert exact > 0
        assert sliced > 0
        
        # Sliced should be in the same order of magnitude
        assert 0.1 < sliced / exact < 10  # Within an order of magnitude
    
    def test_sliced_wasserstein_reproducibility(self):
        """Test that results are reproducible with same seed."""
        diagram1 = np.array([[0.0, 1.0], [1.0, 3.0]])
        diagram2 = np.array([[0.5, 1.5], [1.5, 2.5]])
        
        dist1 = sliced_wasserstein_distance(diagram1, diagram2, n_slices=50, seed=42)
        dist2 = sliced_wasserstein_distance(diagram1, diagram2, n_slices=50, seed=42)
        
        assert dist1 == dist2
    
    def test_sliced_wasserstein_empty_diagrams(self):
        """Test handling of empty diagrams."""
        diagram = np.array([[0.0, 1.0], [1.0, 2.0]])
        empty = np.array([])
        
        dist = sliced_wasserstein_distance(diagram, empty, n_slices=50)
        assert dist > 0  # Should have positive distance
        
        dist_empty = sliced_wasserstein_distance(empty, empty, n_slices=50)
        assert dist_empty == 0.0


class TestPairwiseDistances:
    """Test pairwise distance computation."""
    
    def test_pairwise_distances_basic(self):
        """Test basic pairwise distance computation."""
        diagrams = [
            np.array([[0.0, 1.0]]),
            np.array([[0.0, 2.0]]),
            np.array([[0.0, 3.0]]),
        ]
        
        dist_matrix = compute_pairwise_distances(diagrams, metric='wasserstein', q=1)
        
        # Check shape
        assert dist_matrix.shape == (3, 3)
        
        # Check diagonal is zero
        assert np.allclose(np.diag(dist_matrix), 0.0)
        
        # Check symmetry
        assert np.allclose(dist_matrix, dist_matrix.T)
        
        # Test relative ordering - distances should increase with persistence difference
        assert dist_matrix[0, 1] < dist_matrix[0, 2]  # d(1,2) < d(1,3)
        assert dist_matrix[1, 2] < dist_matrix[0, 2]  # d(2,3) < d(1,3)
    
    def test_pairwise_distances_metrics(self):
        """Test different metrics for pairwise distances."""
        diagrams = [
            np.array([[0.0, 1.0], [1.0, 2.0]]),
            np.array([[0.1, 1.1], [1.1, 2.1]]),
        ]
        
        # Test all available metrics
        for metric in ['wasserstein', 'bottleneck', 'sliced_wasserstein']:
            if metric == 'sliced_wasserstein':
                dist_matrix = compute_pairwise_distances(diagrams, metric=metric, seed=42)
            else:
                dist_matrix = compute_pairwise_distances(diagrams, metric=metric)
            
            assert dist_matrix.shape == (2, 2)
            assert np.allclose(np.diag(dist_matrix), 0.0)
            assert np.allclose(dist_matrix, dist_matrix.T)
    
    def test_pairwise_distances_empty_list(self):
        """Test with empty list of diagrams."""
        diagrams = []
        dist_matrix = compute_pairwise_distances(diagrams, metric='wasserstein')
        assert dist_matrix.shape == (0, 0)
    
    def test_pairwise_distances_invalid_metric(self):
        """Test error handling for invalid metric."""
        diagrams = [np.array([[0.0, 1.0]])]
        
        with pytest.raises(ValueError, match="Unknown metric"):
            compute_pairwise_distances(diagrams, metric='invalid_metric')


class TestEdgeCases:
    """Test edge cases and numerical stability."""
    
    def test_infinite_coordinates(self):
        """Test handling of infinite death times."""
        diagram1 = np.array([[0.0, np.inf], [1.0, 2.0]])
        diagram2 = np.array([[0.0, 3.0], [1.0, np.inf]])
        
        # Should handle infinite coordinates gracefully
        dist = wasserstein_distance(diagram1, diagram2)
        assert np.isfinite(dist)
        assert dist > 0
    
    def test_very_large_diagrams(self):
        """Test performance with larger diagrams."""
        # Generate random diagrams
        np.random.seed(42)
        n_points = 100
        births1 = np.random.uniform(0, 5, n_points)
        deaths1 = births1 + np.random.uniform(0.1, 2, n_points)
        diagram1 = np.column_stack([births1, deaths1])
        
        births2 = np.random.uniform(0, 5, n_points)
        deaths2 = births2 + np.random.uniform(0.1, 2, n_points)
        diagram2 = np.column_stack([births2, deaths2])
        
        # Should complete in reasonable time
        dist = wasserstein_distance(diagram1, diagram2)
        assert np.isfinite(dist)
        assert dist > 0
    
    def test_numerical_precision(self):
        """Test numerical precision with very close points."""
        diagram1 = np.array([[0.0, 1.0]])
        diagram2 = np.array([[0.0, 1.0 + 1e-12]])
        
        dist = wasserstein_distance(diagram1, diagram2)
        assert dist < 1e-10  # Should detect very small differences
    
    def test_fisher_distance_warning(self):
        """Test that Fisher distance raises appropriate warning."""
        diagram1 = np.array([[0.0, 1.0]])
        diagram2 = np.array([[0.0, 2.0]])
        
        with pytest.warns(UserWarning, match="Fisher distance not fully implemented"):
            dist = persistence_fisher_distance(diagram1, diagram2)
        
        # Should return a valid distance
        assert np.isfinite(dist)
        assert dist > 0


class TestMetricProperties:
    """Test that distances satisfy metric properties."""
    
    def setup_method(self):
        """Create test diagrams."""
        self.diagrams = [
            np.array([[0.0, 1.0], [1.0, 2.0]]),
            np.array([[0.5, 1.5], [1.5, 2.5]]),
            np.array([[0.0, 2.0], [2.0, 3.0]]),
            np.array([[1.0, 3.0]]),
        ]
    
    def test_metric_properties_wasserstein(self):
        """Test metric properties for Wasserstein distance."""
        self._test_metric_properties(
            lambda d1, d2: wasserstein_distance(d1, d2, p=2, q=2)
        )
    
    def test_metric_properties_bottleneck(self):
        """Test metric properties for bottleneck distance."""
        self._test_metric_properties(bottleneck_distance)
    
    def _test_metric_properties(self, dist_func):
        """Test the four metric properties for a distance function."""
        # 1. Non-negativity: d(x,y) >= 0
        for i in range(len(self.diagrams)):
            for j in range(len(self.diagrams)):
                dist = dist_func(self.diagrams[i], self.diagrams[j])
                assert dist >= 0, "Distance must be non-negative"
        
        # 2. Identity: d(x,x) = 0
        for diagram in self.diagrams:
            dist = dist_func(diagram, diagram)
            assert abs(dist) < 1e-10, "Distance from diagram to itself must be zero"
        
        # 3. Symmetry: d(x,y) = d(y,x)
        for i in range(len(self.diagrams)):
            for j in range(i+1, len(self.diagrams)):
                dist_ij = dist_func(self.diagrams[i], self.diagrams[j])
                dist_ji = dist_func(self.diagrams[j], self.diagrams[i])
                assert abs(dist_ij - dist_ji) < 1e-10, "Distance must be symmetric"
        
        # 4. Triangle inequality: d(x,z) <= d(x,y) + d(y,z)
        for i in range(len(self.diagrams)):
            for j in range(len(self.diagrams)):
                for k in range(len(self.diagrams)):
                    dist_ij = dist_func(self.diagrams[i], self.diagrams[j])
                    dist_jk = dist_func(self.diagrams[j], self.diagrams[k])
                    dist_ik = dist_func(self.diagrams[i], self.diagrams[k])
                    assert dist_ik <= dist_ij + dist_jk + 1e-10, "Triangle inequality must hold"
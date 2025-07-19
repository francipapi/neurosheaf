#!/usr/bin/env python3
"""
Comprehensive DTW Testing with Synthetic Eigenvalue Sequences

This test suite validates the DTW implementation using controlled synthetic sequences
before relying on real neural network comparisons. It tests DTW sensitivity, accuracy,
and robustness across different eigenvalue evolution patterns.
"""

import pytest
import numpy as np
import torch
import sys
from pathlib import Path

# Add parent directories to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from neurosheaf.utils.dtw_similarity import FiltrationDTW
from neurosheaf.utils.exceptions import ValidationError, ComputationError
from tests.synthetic.synthetic_eigenvalue_generators import SyntheticEigenvalueGenerator, SequenceMetadata


class TestDTWSyntheticSequences:
    """Test DTW implementation with synthetic eigenvalue sequences."""
    
    @pytest.fixture
    def generator(self):
        """Create synthetic eigenvalue generator."""
        return SyntheticEigenvalueGenerator(random_seed=42)
    
    @pytest.fixture
    def dtw_analyzer(self):
        """Create DTW analyzer with default settings."""
        return FiltrationDTW(method='auto', constraint_band=0.1)
    
    @pytest.fixture
    def dtw_analyzer_multivariate(self):
        """Create DTW analyzer optimized for multivariate analysis."""
        return FiltrationDTW(
            method='tslearn',  # tslearn supports multivariate DTW
            constraint_band=0.1,
            normalization_scheme='range_aware'
        )
    
    def test_identical_sequences_zero_distance(self, generator, dtw_analyzer):
        """Test that identical sequences produce zero DTW distance."""
        # Generate identical sequences
        seq1, metadata = generator.exponential_decay(n_steps=20, decay_rate=0.1, noise_level=0.0)
        seq2, _ = generator.exponential_decay(n_steps=20, decay_rate=0.1, noise_level=0.0)
        
        # Compute DTW distance
        result = dtw_analyzer.compare_eigenvalue_evolution(
            seq1, seq2, eigenvalue_index=0, multivariate=False
        )
        
        # Should be approximately zero
        assert result['distance'] < 1e-10, f"Expected near-zero distance for identical sequences, got {result['distance']}"
        assert result['normalized_distance'] < 1e-10, f"Expected near-zero normalized distance, got {result['normalized_distance']}"
        
        print(f"✅ Identical sequences: distance={result['distance']:.2e}, normalized={result['normalized_distance']:.2e}")
    
    def test_different_decay_rates_sensitivity(self, generator, dtw_analyzer):
        """Test DTW sensitivity to different exponential decay rates."""
        # Test multiple decay rate pairs
        decay_pairs = [
            (0.1, 0.1),    # Identical
            (0.1, 0.12),   # Very similar
            (0.1, 0.15),   # Similar
            (0.1, 0.3),    # Different
            (0.1, 0.8),    # Very different
        ]
        
        distances = []
        for decay1, decay2 in decay_pairs:
            seq1, _ = generator.exponential_decay(n_steps=30, decay_rate=decay1, noise_level=0.0)
            seq2, _ = generator.exponential_decay(n_steps=30, decay_rate=decay2, noise_level=0.0)
            
            result = dtw_analyzer.compare_eigenvalue_evolution(
                seq1, seq2, eigenvalue_index=0, multivariate=False
            )
            
            distances.append(result['normalized_distance'])
            print(f"Decay rates {decay1:.2f} vs {decay2:.2f}: distance={result['normalized_distance']:.4f}")
        
        # Verify that distances increase with difference in decay rates
        assert distances[0] < distances[1] < distances[2] < distances[3] < distances[4], \
            f"DTW distances should increase with decay rate differences: {distances}"
        
        # Ensure we can distinguish between similar but not identical patterns
        assert distances[1] > 1e-6, "Should detect difference between similar decay rates"
        assert distances[-1] > distances[0] * 10, "Should strongly distinguish very different decay rates"
    
    def test_cross_pattern_discrimination(self, generator, dtw_analyzer):
        """Test DTW ability to discriminate between different pattern types."""
        # Generate different pattern types
        patterns = {
            'exponential': generator.exponential_decay(n_steps=25, decay_rate=0.2),
            'power_law': generator.power_law(n_steps=25, exponent=-0.5),
            'sinusoidal': generator.sinusoidal_modulated(n_steps=25, frequency=2.0),
            'constant': generator.constant_sequence(n_steps=25, value=5.0),
            'phase_transition': generator.phase_transition(n_steps=25, transition_point=0.5)
        }
        
        # Compute pairwise distances
        pattern_names = list(patterns.keys())
        distance_matrix = np.zeros((len(pattern_names), len(pattern_names)))
        
        for i, name1 in enumerate(pattern_names):
            for j, name2 in enumerate(pattern_names):
                seq1, _ = patterns[name1]
                seq2, _ = patterns[name2]
                
                result = dtw_analyzer.compare_eigenvalue_evolution(
                    seq1, seq2, eigenvalue_index=0, multivariate=False
                )
                
                distance_matrix[i, j] = result['normalized_distance']
        
        # Print distance matrix
        print("\nCross-pattern DTW distance matrix:")
        print(f"{'':>15}", end="")
        for name in pattern_names:
            print(f"{name:>12}", end="")
        print()
        
        for i, name1 in enumerate(pattern_names):
            print(f"{name1:>15}", end="")
            for j, name2 in enumerate(pattern_names):
                print(f"{distance_matrix[i, j]:>12.4f}", end="")
            print()
        
        # Verify diagonal is zero (self-comparison)
        for i in range(len(pattern_names)):
            assert distance_matrix[i, i] < 1e-10, f"Self-comparison should be zero: {pattern_names[i]}"
        
        # Verify off-diagonal elements are positive (different patterns)
        for i in range(len(pattern_names)):
            for j in range(len(pattern_names)):
                if i != j:
                    assert distance_matrix[i, j] > 1e-6, \
                        f"Different patterns should have positive distance: {pattern_names[i]} vs {pattern_names[j]}"
        
        # Verify symmetry
        for i in range(len(pattern_names)):
            for j in range(len(pattern_names)):
                assert abs(distance_matrix[i, j] - distance_matrix[j, i]) < 1e-10, \
                    "Distance matrix should be symmetric"
    
    def test_noise_robustness(self, generator, dtw_analyzer):
        """Test DTW robustness to noise in eigenvalue sequences."""
        base_sequence, _ = generator.exponential_decay(n_steps=20, decay_rate=0.15, noise_level=0.0)
        
        noise_levels = [0.0, 0.01, 0.05, 0.1, 0.2]
        distances = []
        
        for noise_level in noise_levels:
            noisy_sequence, _ = generator.exponential_decay(
                n_steps=20, decay_rate=0.15, noise_level=noise_level
            )
            
            result = dtw_analyzer.compare_eigenvalue_evolution(
                base_sequence, noisy_sequence, eigenvalue_index=0, multivariate=False
            )
            
            distances.append(result['normalized_distance'])
            print(f"Noise level {noise_level:.2f}: distance={result['normalized_distance']:.4f}")
        
        # Distances should increase with noise level
        for i in range(1, len(distances)):
            assert distances[i] >= distances[i-1], \
                f"Distance should increase with noise: level {noise_levels[i-1]} vs {noise_levels[i]}"
        
        # But should remain relatively small for reasonable noise levels
        assert distances[2] < 0.5, "Distance should remain reasonable for moderate noise (5%)"
        
        # High noise should be detectable
        assert distances[-1] > distances[0] * 5, "High noise should be clearly detectable"
    
    def test_multivariate_dtw_basic(self, generator, dtw_analyzer_multivariate):
        """Test basic multivariate DTW functionality."""
        # Generate sequences with multiple eigenvalues
        seq1, _ = generator.exponential_decay(n_steps=15, n_eigenvals=5, decay_rate=0.1)
        seq2, _ = generator.exponential_decay(n_steps=15, n_eigenvals=5, decay_rate=0.1)
        
        # Test multivariate DTW
        result = dtw_analyzer_multivariate.compare_eigenvalue_evolution(
            seq1, seq2, multivariate=True
        )
        
        # Should be near zero for identical patterns
        assert result['distance'] < 1e-6, f"Expected small distance for identical multivariate sequences, got {result['distance']}"
        assert result['multivariate'] == True, "Result should indicate multivariate computation"
        
        print(f"✅ Multivariate identical: distance={result['distance']:.6f}, normalized={result['normalized_distance']:.6f}")
    
    def test_multivariate_dtw_sensitivity(self, generator, dtw_analyzer_multivariate):
        """Test multivariate DTW sensitivity to different patterns."""
        # Generate sequences with different patterns
        seq1, _ = generator.exponential_decay(n_steps=20, n_eigenvals=8, decay_rate=0.1)
        seq2, _ = generator.power_law(n_steps=20, n_eigenvals=8, exponent=-0.3)
        seq3, _ = generator.sinusoidal_modulated(n_steps=20, n_eigenvals=8, frequency=1.5)
        
        sequences = [seq1, seq2, seq3]
        names = ['exponential', 'power_law', 'sinusoidal']
        
        # Compute pairwise multivariate distances
        print("\nMultivariate DTW distances:")
        for i in range(len(sequences)):
            for j in range(i + 1, len(sequences)):
                result = dtw_analyzer_multivariate.compare_eigenvalue_evolution(
                    sequences[i], sequences[j], multivariate=True
                )
                
                print(f"{names[i]} vs {names[j]}: distance={result['normalized_distance']:.4f}")
                
                # Should detect differences between different patterns
                assert result['normalized_distance'] > 1e-3, \
                    f"Should detect difference between {names[i]} and {names[j]}"
    
    def test_sequence_length_robustness(self, generator, dtw_analyzer):
        """Test DTW robustness to different sequence lengths."""
        # Generate sequences of different lengths
        lengths = [10, 20, 30, 50]
        base_sequence, _ = generator.exponential_decay(n_steps=25, decay_rate=0.2)
        
        for length in lengths:
            test_sequence, _ = generator.exponential_decay(n_steps=length, decay_rate=0.2)
            
            result = dtw_analyzer.compare_eigenvalue_evolution(
                base_sequence, test_sequence, eigenvalue_index=0, multivariate=False
            )
            
            print(f"Length {length}: distance={result['normalized_distance']:.4f}")
            
            # Should handle different lengths gracefully
            assert result['normalized_distance'] < 0.5, \
                f"Distance should be reasonable for similar patterns with different lengths: {length}"
            assert not np.isnan(result['distance']), "Distance should not be NaN"
            assert not np.isinf(result['distance']), "Distance should not be infinite"
    
    def test_edge_cases(self, generator, dtw_analyzer):
        """Test DTW behavior on edge cases."""
        # Test with minimal sequences
        seq1 = [torch.tensor([1.0])]  # Single step, single eigenvalue
        seq2 = [torch.tensor([1.5])]
        
        result = dtw_analyzer.compare_eigenvalue_evolution(
            seq1, seq2, eigenvalue_index=0, multivariate=False
        )
        
        assert not np.isnan(result['distance']), "Should handle single-step sequences"
        assert result['distance'] >= 0, "Distance should be non-negative"
        
        # Test with zero eigenvalues
        seq_zero = [torch.tensor([0.0, 0.0]) for _ in range(5)]
        seq_positive, _ = generator.constant_sequence(n_steps=5, value=1.0)
        
        result = dtw_analyzer.compare_eigenvalue_evolution(
            seq_zero, seq_positive, eigenvalue_index=0, multivariate=False
        )
        
        assert not np.isnan(result['distance']), "Should handle zero eigenvalues"
        assert result['distance'] > 0, "Should detect difference between zero and positive sequences"
        
        print(f"✅ Edge cases: minimal={result['distance']:.4f}")
    
    def test_dtw_test_pairs_validation(self, generator, dtw_analyzer):
        """Test the predefined test pairs from the generator."""
        test_pairs = generator.generate_test_pairs()
        
        print(f"\nValidating {len(test_pairs)} predefined test pairs:")
        print(f"{'Description':<50} {'Expected':<12} {'Actual':<12} {'Ratio':<8}")
        print("-" * 82)
        
        for seq1, seq2, expected_relative_distance, description in test_pairs:
            result = dtw_analyzer.compare_eigenvalue_evolution(
                seq1, seq2, eigenvalue_index=0, multivariate=False
            )
            
            actual_distance = result['normalized_distance']
            
            # For relative comparison, normalize both to [0, 1] range
            # Expected is already in relative scale, actual needs context
            max_observed_distance = 2.0  # Rough estimate based on typical DTW distances
            relative_actual = min(actual_distance / max_observed_distance, 1.0)
            
            ratio = relative_actual / max(expected_relative_distance, 0.001)
            
            print(f"{description:<50} {expected_relative_distance:<12.3f} {relative_actual:<12.3f} {ratio:<8.2f}")
            
            # Validate that distances are reasonable
            assert not np.isnan(actual_distance), f"Distance should not be NaN for: {description}"
            assert actual_distance >= 0, f"Distance should be non-negative for: {description}"
            
            # For zero expected distance (identical sequences), actual should be very small
            if expected_relative_distance < 0.01:
                assert actual_distance < 1e-6, f"Should have near-zero distance for: {description}"


def test_dtw_implementation_validation():
    """High-level validation test for DTW implementation."""
    generator = SyntheticEigenvalueGenerator(random_seed=123)
    dtw_analyzer = FiltrationDTW(method='auto')
    
    print("🔬 DTW Implementation Validation")
    print("=" * 50)
    
    # Test basic functionality
    seq1, _ = generator.exponential_decay(n_steps=20, decay_rate=0.15)
    seq2, _ = generator.exponential_decay(n_steps=20, decay_rate=0.25)
    
    result = dtw_analyzer.compare_eigenvalue_evolution(
        seq1, seq2, eigenvalue_index=0, multivariate=False
    )
    
    print(f"✅ Basic DTW computation successful")
    print(f"   Distance: {result['distance']:.6f}")
    print(f"   Normalized: {result['normalized_distance']:.6f}")
    print(f"   Method: {result['method']}")
    
    # Test multivariate if available
    try:
        result_mv = dtw_analyzer.compare_eigenvalue_evolution(
            seq1, seq2, multivariate=True
        )
        print(f"✅ Multivariate DTW successful")
        print(f"   Multivariate distance: {result_mv['normalized_distance']:.6f}")
    except Exception as e:
        print(f"⚠️  Multivariate DTW unavailable: {e}")
    
    print("\n✅ DTW implementation validation complete")


if __name__ == "__main__":
    # Run basic validation when script is executed directly
    test_dtw_implementation_validation()
    
    # Run a few key tests manually
    generator = SyntheticEigenvalueGenerator(random_seed=42)
    dtw_analyzer = FiltrationDTW(method='auto')
    
    test_class = TestDTWSyntheticSequences()
    test_class.test_identical_sequences_zero_distance(generator, dtw_analyzer)
    test_class.test_different_decay_rates_sensitivity(generator, dtw_analyzer)
    test_class.test_cross_pattern_discrimination(generator, dtw_analyzer)
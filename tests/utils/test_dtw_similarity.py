"""Tests for DTW similarity functionality."""

import pytest
import torch
import numpy as np
from unittest.mock import Mock, patch
from neurosheaf.utils.dtw_similarity import FiltrationDTW, create_filtration_dtw_comparator, quick_dtw_comparison
from neurosheaf.utils.exceptions import ValidationError, ComputationError


class TestFiltrationDTW:
    """Test suite for FiltrationDTW class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.dtw = FiltrationDTW(method='auto')
        
        # Create sample eigenvalue sequences
        self.eigenvals1 = [
            torch.tensor([1.0, 0.5, 0.1]),
            torch.tensor([0.8, 0.4]),
            torch.tensor([0.6])
        ]
        
        self.eigenvals2 = [
            torch.tensor([1.2, 0.6, 0.2]),
            torch.tensor([0.9, 0.5, 0.1]),
            torch.tensor([0.7, 0.3])
        ]
        
        self.filtration_params1 = [0.1, 0.2, 0.3]
        self.filtration_params2 = [0.1, 0.2, 0.3]
    
    def test_init_valid_parameters(self):
        """Test FiltrationDTW initialization with valid parameters."""
        dtw = FiltrationDTW(
            method='auto',
            constraint_band=0.1,
            eigenvalue_weight=0.7,
            structural_weight=0.3,
            min_eigenvalue_threshold=1e-10
        )
        
        assert dtw.constraint_band == 0.1
        assert dtw.eigenvalue_weight == 0.7
        assert dtw.structural_weight == 0.3
        assert dtw.min_eigenvalue_threshold == 1e-10
    
    def test_init_invalid_weights(self):
        """Test initialization with invalid weight parameters."""
        with pytest.raises(ValidationError, match="eigenvalue_weight \\+ structural_weight must equal 1.0"):
            FiltrationDTW(eigenvalue_weight=0.5, structural_weight=0.4)
    
    def test_init_invalid_constraint_band(self):
        """Test initialization with invalid constraint band."""
        with pytest.raises(ValidationError, match="constraint_band must be between 0.0 and 1.0"):
            FiltrationDTW(constraint_band=1.5)
    
    def test_select_method_auto(self):
        """Test automatic method selection."""
        dtw = FiltrationDTW(method='auto')
        # Should select an available method
        assert dtw.method in ['dtaidistance', 'tslearn', 'dtw-python']
    
    def test_select_method_unavailable(self):
        """Test error when requested method is unavailable."""
        with patch('neurosheaf.utils.dtw_similarity.DTW_AVAILABLE', False):
            with patch('neurosheaf.utils.dtw_similarity.TSLEARN_AVAILABLE', False):
                with patch('neurosheaf.utils.dtw_similarity.DTW_PYTHON_AVAILABLE', False):
                    with pytest.raises(ComputationError, match="No DTW libraries available"):
                        FiltrationDTW(method='auto')
    
    def test_validate_evolution_sequences_valid(self):
        """Test validation of valid evolution sequences."""
        # Should not raise exception
        self.dtw._validate_evolution_sequences(self.eigenvals1, self.eigenvals2)
    
    def test_validate_evolution_sequences_empty(self):
        """Test validation with empty sequences."""
        with pytest.raises(ValidationError, match="Evolution sequences cannot be empty"):
            self.dtw._validate_evolution_sequences([], self.eigenvals2)
    
    def test_validate_evolution_sequences_non_tensor(self):
        """Test validation with non-tensor elements."""
        invalid_seq = [torch.tensor([1.0]), [1.0, 2.0]]  # Mixed types
        with pytest.raises(ValidationError, match="is not a tensor"):
            self.dtw._validate_evolution_sequences(invalid_seq, self.eigenvals2)
    
    def test_extract_univariate_sequences(self):
        """Test extraction of univariate sequences."""
        seq1, seq2 = self.dtw._extract_univariate_sequences(
            self.eigenvals1, self.eigenvals2, eigenvalue_index=0
        )
        
        assert len(seq1) == 3
        assert len(seq2) == 3
        assert seq1[0] == 1.0  # First eigenvalue from first sequence
        assert seq2[0] == 1.2  # First eigenvalue from second sequence
    
    def test_extract_univariate_sequences_missing_eigenvalue(self):
        """Test extraction when eigenvalue index is missing."""
        seq1, seq2 = self.dtw._extract_univariate_sequences(
            self.eigenvals1, self.eigenvals2, eigenvalue_index=5  # Non-existent index
        )
        
        # Should use threshold value for missing eigenvalues
        assert np.all(seq1 == self.dtw.min_eigenvalue_threshold)
        assert np.all(seq2 == self.dtw.min_eigenvalue_threshold)
    
    def test_extract_multivariate_sequences(self):
        """Test extraction of multivariate sequences."""
        seq1, seq2 = self.dtw._extract_multivariate_sequences(
            self.eigenvals1, self.eigenvals2
        )
        
        assert seq1.shape[0] == 3  # Number of filtration steps
        assert seq2.shape[0] == 3  # Number of filtration steps
        assert seq1.shape[1] == seq2.shape[1]  # Same number of eigenvalues (padded)
    
    def test_compare_eigenvalue_evolution_univariate(self):
        """Test univariate eigenvalue evolution comparison."""
        result = self.dtw.compare_eigenvalue_evolution(
            self.eigenvals1, self.eigenvals2,
            self.filtration_params1, self.filtration_params2,
            eigenvalue_index=0, multivariate=False
        )
        
        assert 'distance' in result
        assert 'normalized_distance' in result
        assert 'alignment' in result
        assert 'alignment_visualization' in result
        assert isinstance(result['distance'], float)
        assert result['distance'] >= 0
    
    def test_compare_eigenvalue_evolution_multivariate(self):
        """Test multivariate eigenvalue evolution comparison."""
        with patch('neurosheaf.utils.dtw_similarity.TSLEARN_AVAILABLE', True):
            result = self.dtw.compare_eigenvalue_evolution(
                self.eigenvals1, self.eigenvals2,
                self.filtration_params1, self.filtration_params2,
                multivariate=True
            )
            
            assert 'distance' in result
            assert 'multivariate' in result
            assert result['multivariate'] == True
    
    def test_compare_multiple_evolutions(self):
        """Test comparison of multiple eigenvalue evolutions."""
        evolutions = [self.eigenvals1, self.eigenvals2]
        filtration_params = [self.filtration_params1, self.filtration_params2]
        
        distance_matrix = self.dtw.compare_multiple_evolutions(
            evolutions, filtration_params, eigenvalue_index=0
        )
        
        assert distance_matrix.shape == (2, 2)
        assert distance_matrix[0, 0] == 0.0  # Self-distance should be 0
        assert distance_matrix[0, 1] == distance_matrix[1, 0]  # Symmetric
    
    def test_prepare_alignment_visualization(self):
        """Test preparation of alignment visualization data."""
        seq1 = np.array([1.0, 0.8, 0.6])
        seq2 = np.array([1.2, 0.9, 0.7])
        alignment = [(0, 0), (1, 1), (2, 2)]
        
        viz_data = self.dtw._prepare_alignment_visualization(
            seq1, seq2, alignment, self.filtration_params1, self.filtration_params2
        )
        
        assert 'sequence1' in viz_data
        assert 'sequence2' in viz_data
        assert 'alignment' in viz_data
        assert 'alignment_quality' in viz_data
        assert len(viz_data['sequence1']) == 3
        assert len(viz_data['sequence2']) == 3
    
    @patch('neurosheaf.utils.dtw_similarity.DTW_AVAILABLE', True)
    @patch('neurosheaf.utils.dtw_similarity.dtw')
    def test_dtaidistance_univariate(self, mock_dtw):
        """Test DTW computation using dtaidistance library."""
        mock_dtw.distance_fast.return_value = 0.5
        
        seq1 = np.array([1.0, 0.8, 0.6])
        seq2 = np.array([1.2, 0.9, 0.7])
        
        distance, alignment = self.dtw._dtaidistance_univariate(seq1, seq2)
        
        assert distance == 0.5
        assert isinstance(alignment, list)
        mock_dtw.distance_fast.assert_called_once()
    
    @patch('neurosheaf.utils.dtw_similarity.TSLEARN_AVAILABLE', True)
    @patch('neurosheaf.utils.dtw_similarity.ts_dtw')
    @patch('neurosheaf.utils.dtw_similarity.dtw_path')
    def test_tslearn_univariate(self, mock_dtw_path, mock_ts_dtw):
        """Test DTW computation using tslearn library."""
        mock_ts_dtw.return_value = 0.3
        mock_dtw_path.return_value = ([(0, 0), (1, 1), (2, 2)], 0.3)
        
        seq1 = np.array([1.0, 0.8, 0.6])
        seq2 = np.array([1.2, 0.9, 0.7])
        
        distance, alignment = self.dtw._tslearn_univariate(seq1, seq2)
        
        assert distance == 0.3
        assert len(alignment) == 3
        mock_ts_dtw.assert_called_once()
        mock_dtw_path.assert_called_once()
    
    @patch('neurosheaf.utils.dtw_similarity.TSLEARN_AVAILABLE', True)
    @patch('neurosheaf.utils.dtw_similarity.ts_dtw')
    @patch('neurosheaf.utils.dtw_similarity.dtw_path')
    def test_tslearn_multivariate(self, mock_dtw_path, mock_ts_dtw):
        """Test multivariate DTW computation using tslearn."""
        mock_ts_dtw.return_value = 0.4
        mock_dtw_path.return_value = ([(0, 0), (1, 1), (2, 2)], 0.4)
        
        seq1 = np.array([[1.0, 0.5], [0.8, 0.4], [0.6, 0.3]])
        seq2 = np.array([[1.2, 0.6], [0.9, 0.5], [0.7, 0.4]])
        
        distance, alignment = self.dtw._tslearn_multivariate(seq1, seq2)
        
        assert distance == 0.4
        assert len(alignment) == 3
        mock_ts_dtw.assert_called_once()
        mock_dtw_path.assert_called_once()
    
    def test_compute_dtw_unknown_method(self):
        """Test DTW computation with unknown method."""
        dtw = FiltrationDTW(method='auto')
        dtw.method = 'unknown_method'
        
        seq1 = np.array([1.0, 0.8, 0.6])
        seq2 = np.array([1.2, 0.9, 0.7])
        
        with pytest.raises(ComputationError, match="Unknown DTW method"):
            dtw._compute_univariate_dtw(seq1, seq2)


class TestFactoryFunctions:
    """Test factory functions and utility functions."""
    
    def test_create_filtration_dtw_comparator(self):
        """Test factory function for creating DTW comparator."""
        comparator = create_filtration_dtw_comparator(method='auto')
        
        assert isinstance(comparator, FiltrationDTW)
        assert comparator.method in ['dtaidistance', 'tslearn', 'dtw-python']
    
    def test_create_filtration_dtw_comparator_with_kwargs(self):
        """Test factory function with additional kwargs."""
        comparator = create_filtration_dtw_comparator(
            method='auto',
            constraint_band=0.2,
            eigenvalue_weight=0.8,
            structural_weight=0.2
        )
        
        assert comparator.constraint_band == 0.2
        assert comparator.eigenvalue_weight == 0.8
        assert comparator.structural_weight == 0.2
    
    def test_quick_dtw_comparison(self):
        """Test quick DTW comparison function."""
        eigenvals1 = [torch.tensor([1.0, 0.5]), torch.tensor([0.8])]
        eigenvals2 = [torch.tensor([1.2, 0.6]), torch.tensor([0.9])]
        
        distance = quick_dtw_comparison(eigenvals1, eigenvals2, eigenvalue_index=0)
        
        assert isinstance(distance, float)
        assert distance >= 0


class TestDTWIntegration:
    """Integration tests for DTW functionality."""
    
    def test_dtw_with_real_eigenvalue_sequences(self):
        """Test DTW with realistic eigenvalue sequences."""
        # Create more realistic eigenvalue sequences
        n_steps = 20
        eigenvals1 = []
        eigenvals2 = []
        
        for i in range(n_steps):
            # First network: exponential decay
            vals1 = torch.tensor([
                np.exp(-0.1 * i),
                np.exp(-0.2 * i),
                np.exp(-0.3 * i)
            ])
            eigenvals1.append(vals1)
            
            # Second network: power law decay (similar but different pattern)
            vals2 = torch.tensor([
                (i + 1) ** -0.5,
                (i + 1) ** -0.8,
                (i + 1) ** -1.2
            ])
            eigenvals2.append(vals2)
        
        dtw = FiltrationDTW(method='auto')
        
        # Compare univariate sequences
        result = dtw.compare_eigenvalue_evolution(
            eigenvals1, eigenvals2,
            eigenvalue_index=0, multivariate=False
        )
        
        assert result['distance'] > 0
        assert result['normalized_distance'] > 0
        assert len(result['alignment_visualization']['sequence1']) == n_steps
        assert len(result['alignment_visualization']['sequence2']) == n_steps
    
    def test_dtw_with_different_sequence_lengths(self):
        """Test DTW with sequences of different lengths."""
        eigenvals1 = [torch.tensor([1.0]), torch.tensor([0.5])]
        eigenvals2 = [torch.tensor([1.2]), torch.tensor([0.8]), torch.tensor([0.3])]
        
        dtw = FiltrationDTW(method='auto')
        
        result = dtw.compare_eigenvalue_evolution(
            eigenvals1, eigenvals2,
            eigenvalue_index=0, multivariate=False
        )
        
        assert result['distance'] >= 0
        assert result['sequence1_length'] == 2
        assert result['sequence2_length'] == 3
    
    def test_dtw_with_constraint_band(self):
        """Test DTW with constraint band applied."""
        eigenvals1 = [torch.tensor([1.0, 0.5]), torch.tensor([0.8, 0.4])]
        eigenvals2 = [torch.tensor([1.2, 0.6]), torch.tensor([0.9, 0.5])]
        
        dtw_constrained = FiltrationDTW(method='auto', constraint_band=0.1)
        dtw_unconstrained = FiltrationDTW(method='auto', constraint_band=0.0)
        
        result_constrained = dtw_constrained.compare_eigenvalue_evolution(
            eigenvals1, eigenvals2, eigenvalue_index=0
        )
        result_unconstrained = dtw_unconstrained.compare_eigenvalue_evolution(
            eigenvals1, eigenvals2, eigenvalue_index=0
        )
        
        # Both should produce valid results
        assert result_constrained['distance'] >= 0
        assert result_unconstrained['distance'] >= 0
    
    def test_dtw_performance_with_large_sequences(self):
        """Test DTW performance with larger sequences."""
        n_steps = 100
        eigenvals1 = [torch.tensor([np.random.random()]) for _ in range(n_steps)]
        eigenvals2 = [torch.tensor([np.random.random()]) for _ in range(n_steps)]
        
        dtw = FiltrationDTW(method='auto')
        
        # This should complete without timing out
        result = dtw.compare_eigenvalue_evolution(
            eigenvals1, eigenvals2,
            eigenvalue_index=0, multivariate=False
        )
        
        assert result['distance'] >= 0
        assert result['sequence1_length'] == n_steps
        assert result['sequence2_length'] == n_steps


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_empty_eigenvalue_tensors(self):
        """Test handling of empty eigenvalue tensors."""
        eigenvals1 = [torch.tensor([]), torch.tensor([1.0])]
        eigenvals2 = [torch.tensor([1.0]), torch.tensor([])]
        
        dtw = FiltrationDTW(method='auto')
        
        result = dtw.compare_eigenvalue_evolution(
            eigenvals1, eigenvals2,
            eigenvalue_index=0, multivariate=False
        )
        
        # Should handle empty tensors gracefully
        assert result['distance'] >= 0
    
    def test_single_step_sequences(self):
        """Test with single-step sequences."""
        eigenvals1 = [torch.tensor([1.0, 0.5])]
        eigenvals2 = [torch.tensor([1.2, 0.6])]
        
        dtw = FiltrationDTW(method='auto')
        
        result = dtw.compare_eigenvalue_evolution(
            eigenvals1, eigenvals2,
            eigenvalue_index=0, multivariate=False
        )
        
        assert result['distance'] >= 0
        assert result['sequence1_length'] == 1
        assert result['sequence2_length'] == 1
    
    def test_identical_sequences(self):
        """Test with identical sequences."""
        eigenvals = [torch.tensor([1.0, 0.5]), torch.tensor([0.8, 0.4])]
        
        dtw = FiltrationDTW(method='auto')
        
        result = dtw.compare_eigenvalue_evolution(
            eigenvals, eigenvals,
            eigenvalue_index=0, multivariate=False
        )
        
        # Distance should be 0 for identical sequences
        assert result['distance'] == 0.0
        assert result['normalized_distance'] == 0.0
    
    def test_very_small_eigenvalues(self):
        """Test with very small eigenvalues near threshold."""
        eigenvals1 = [torch.tensor([1e-15, 1e-14])]
        eigenvals2 = [torch.tensor([1e-16, 1e-13])]
        
        dtw = FiltrationDTW(method='auto', min_eigenvalue_threshold=1e-12)
        
        result = dtw.compare_eigenvalue_evolution(
            eigenvals1, eigenvals2,
            eigenvalue_index=0, multivariate=False
        )
        
        # Should handle small values using threshold
        assert result['distance'] >= 0
    
    def test_nan_eigenvalues(self):
        """Test handling of NaN eigenvalues."""
        eigenvals1 = [torch.tensor([1.0, float('nan')])]
        eigenvals2 = [torch.tensor([1.2, 0.6])]
        
        dtw = FiltrationDTW(method='auto')
        
        # Should handle NaN values gracefully or raise appropriate error
        try:
            result = dtw.compare_eigenvalue_evolution(
                eigenvals1, eigenvals2,
                eigenvalue_index=0, multivariate=False
            )
            assert result['distance'] >= 0
        except (ValueError, ComputationError):
            # Either handling NaN gracefully or raising appropriate error is acceptable
            pass


@pytest.mark.slow
class TestPerformance:
    """Performance tests for DTW functionality."""
    
    def test_large_sequence_performance(self):
        """Test performance with large sequences."""
        n_steps = 500
        eigenvals1 = [torch.tensor([np.random.random() * (1.0 - 0.001 * i)]) for i in range(n_steps)]
        eigenvals2 = [torch.tensor([np.random.random() * (1.0 - 0.002 * i)]) for i in range(n_steps)]
        
        dtw = FiltrationDTW(method='auto', constraint_band=0.1)  # Use constraints for performance
        
        import time
        start_time = time.time()
        
        result = dtw.compare_eigenvalue_evolution(
            eigenvals1, eigenvals2,
            eigenvalue_index=0, multivariate=False
        )
        
        end_time = time.time()
        
        # Should complete within reasonable time (less than 10 seconds)
        assert end_time - start_time < 10.0
        assert result['distance'] >= 0
    
    def test_multiple_comparison_performance(self):
        """Test performance of multiple comparisons."""
        n_sequences = 10
        n_steps = 50
        
        evolutions = []
        for _ in range(n_sequences):
            evolution = [torch.tensor([np.random.random()]) for _ in range(n_steps)]
            evolutions.append(evolution)
        
        dtw = FiltrationDTW(method='auto')
        
        import time
        start_time = time.time()
        
        distance_matrix = dtw.compare_multiple_evolutions(evolutions)
        
        end_time = time.time()
        
        # Should complete within reasonable time
        assert end_time - start_time < 30.0
        assert distance_matrix.shape == (n_sequences, n_sequences)
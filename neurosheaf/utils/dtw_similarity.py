"""Dynamic Time Warping for eigenvalue evolution comparison.

This module provides DTW-based similarity measures for comparing eigenvalue
evolution across filtration parameters between different neural networks.
Optimized for the neurosheaf pipeline with efficient library implementations.
"""

import numpy as np
import torch
from typing import Dict, List, Optional, Tuple, Union, Any
import warnings
from ..utils.logging import setup_logger
from ..utils.exceptions import ValidationError, ComputationError

# DTW library imports with fallback handling
try:
    from dtaidistance import dtw
    DTW_AVAILABLE = True
except ImportError:
    DTW_AVAILABLE = False
    warnings.warn("dtaidistance not available. DTW functionality will be limited.")

try:
    from tslearn.metrics import dtw as ts_dtw
    from tslearn.metrics import dtw_path
    TSLEARN_AVAILABLE = True
except ImportError:
    TSLEARN_AVAILABLE = False
    warnings.warn("tslearn not available. Multivariate DTW will be limited.")

try:
    from dtw import dtw as dtw_python
    DTW_PYTHON_AVAILABLE = True
except ImportError:
    DTW_PYTHON_AVAILABLE = False

logger = setup_logger(__name__)


class FiltrationDTW:
    """DTW-based similarity analysis for eigenvalue evolution across filtration.
    
    This class provides methods to compare eigenvalue evolution patterns between
    different neural networks using Dynamic Time Warping, allowing for temporal
    alignment of spectral events during filtration processes.
    
    Features:
    - Univariate DTW for single eigenvalue tracking
    - Multivariate DTW for simultaneous eigenvalue comparison
    - Filtration-aware distance functions
    - Constraint-based alignment (Sakoe-Chiba band)
    - Performance optimizations for large eigenvalue sequences
    
    Attributes:
        method: DTW library to use ('dtaidistance', 'tslearn', 'dtw-python')
        constraint_band: Sakoe-Chiba band constraint ratio (0.0-1.0)
        eigenvalue_weight: Weight for eigenvalue magnitude differences (default: 1.0)
        structural_weight: Weight for structural changes (default: 0.0 for pure functional similarity)
        min_eigenvalue_threshold: Minimum eigenvalue for log-scale computations
        
    Default Configuration:
        Configured for pure functional similarity measurement across different architectures:
        - eigenvalue_weight=1.0: Focus 100% on eigenvalue evolution patterns
        - structural_weight=0.0: Ignore architectural differences (MLP vs CNN vs Custom)
        - This enables fair comparison of functional behavior regardless of model architecture
    """
    
    def __init__(self,
                 method: str = 'auto',
                 constraint_band: float = 0.1,
                 eigenvalue_weight: float = 1.0,
                 structural_weight: float = 0.0,
                 min_eigenvalue_threshold: float = 1e-12,
                 normalization_scheme: str = 'range_aware',
                 zero_sequence_penalty: float = 10.0):
        """Initialize FiltrationDTW.
        
        Args:
            method: DTW implementation ('dtaidistance', 'tslearn', 'dtw-python', 'auto')
            constraint_band: Sakoe-Chiba band constraint (0.0-1.0, 0.0 = no constraint)
            eigenvalue_weight: Weight for eigenvalue magnitude differences (default: 1.0 for pure functional similarity)
            structural_weight: Weight for structural changes (default: 0.0 to ignore architectural differences)
            min_eigenvalue_threshold: Minimum eigenvalue threshold for numerical stability
            
        Note:
            Default configuration (eigenvalue_weight=1.0, structural_weight=0.0) provides pure
            functional similarity measurement, ignoring architectural differences between models.
            This is ideal for comparing functional similarity across different architectures
            (e.g., MLP vs CNN vs Custom models).
        """
        self.method = self._select_method(method)
        self.constraint_band = constraint_band
        self.eigenvalue_weight = eigenvalue_weight
        self.structural_weight = structural_weight
        self.min_eigenvalue_threshold = min_eigenvalue_threshold
        self.normalization_scheme = normalization_scheme
        self.zero_sequence_penalty = zero_sequence_penalty
        
        # Validate weights - allow pure functional similarity
        if eigenvalue_weight < 0.0 or structural_weight < 0.0:
            raise ValidationError("Weights must be non-negative")
        if eigenvalue_weight == 0.0 and structural_weight == 0.0:
            raise ValidationError("At least one weight must be positive")
        
        if not (0.0 <= constraint_band <= 1.0):
            raise ValidationError("constraint_band must be between 0.0 and 1.0")
        
        logger.info(f"FiltrationDTW initialized: method={self.method}, "
                   f"constraint_band={constraint_band}, "
                   f"weights=({eigenvalue_weight}, {structural_weight}) - "
                   f"{'pure functional similarity' if structural_weight == 0.0 else 'balanced functional+structural'}")
    
    def _select_method(self, method: str) -> str:
        """Select appropriate DTW method based on availability."""
        if method == 'auto':
            if DTW_AVAILABLE:
                return 'dtaidistance'
            elif TSLEARN_AVAILABLE:
                return 'tslearn'
            elif DTW_PYTHON_AVAILABLE:
                return 'dtw-python'
            else:
                raise ComputationError("No DTW libraries available. Please install dtaidistance or tslearn.")
        
        # Validate requested method
        if method == 'dtaidistance' and not DTW_AVAILABLE:
            raise ComputationError("dtaidistance not available")
        elif method == 'tslearn' and not TSLEARN_AVAILABLE:
            raise ComputationError("tslearn not available")
        elif method == 'dtw-python' and not DTW_PYTHON_AVAILABLE:
            raise ComputationError("dtw-python not available")
        
        return method
    
    def _compute_enhanced_normalization(self, seq1: np.ndarray, seq2: np.ndarray, raw_distance: float) -> float:
        """Compute enhanced normalization with multivariate awareness and sensitivity preservation."""
        
        # Detect problematic patterns
        issues = self._detect_sequence_issues(seq1, seq2)
        
        # Check if this is multivariate or univariate
        is_multivariate = seq1.ndim > 1 and seq1.shape[1] > 1
        
        if is_multivariate:
            # Multivariate normalization: account for dimensionality and sequence length
            n_features = seq1.shape[1]
            avg_seq_length = (len(seq1) + len(seq2)) / 2
            
            # Scale factor based on sequence properties
            seq1_flat = seq1.flatten()
            seq2_flat = seq2.flatten()
            combined_seq = np.concatenate([seq1_flat, seq2_flat])
            
            # Use range-based normalization for multivariate sequences
            seq_range = np.max(combined_seq) - np.min(combined_seq)
            if seq_range > 1e-12:
                # Normalize by range * sqrt(dimensions) * avg_length to account for multivariate scaling
                base_scale = seq_range * np.sqrt(n_features) * np.sqrt(avg_seq_length) / 10.0
            else:
                base_scale = avg_seq_length  # Fallback for constant sequences
                
            normalized_distance = raw_distance / base_scale
            
        else:
            # Univariate normalization: use robust statistics
            seq1_flat = seq1.flatten() if seq1.ndim > 1 else seq1
            seq2_flat = seq2.flatten() if seq2.ndim > 1 else seq2
            
            # Use median absolute deviation for robust scaling
            combined_seq = np.concatenate([seq1_flat, seq2_flat])
            median_val = np.median(combined_seq)
            mad = np.median(np.abs(combined_seq - median_val))
            
            # Robust scale factor (avoid division by zero)
            if mad > 1e-12:
                scale_factor = mad * 1.4826  # MAD to standard deviation conversion
            else:
                scale_factor = np.std(combined_seq)
                if scale_factor < 1e-12:
                    scale_factor = max(len(seq1), len(seq2))  # Fallback based on sequence length
            
            normalized_distance = raw_distance / scale_factor
        
        # Apply sequence-specific adjustments
        adjusted_normalized = self._apply_normalized_adjustments(normalized_distance, issues)
        
        # Only ensure non-negative result
        final_distance = max(0.0, adjusted_normalized)
        
        # Diagnostic logging for debugging DTW sensitivity issues
        multivar_info = f"multivariate({seq1.shape[1]}D)" if is_multivariate else "univariate"
        logger.debug(f"DTW normalization ({multivar_info}): raw={raw_distance:.4f}, "
                    f"normalized={normalized_distance:.4f}, adjusted={adjusted_normalized:.4f}, "
                    f"final={final_distance:.4f}")
        
        return final_distance
    
    def _detect_sequence_issues(self, seq1: np.ndarray, seq2: np.ndarray) -> Dict[str, float]:
        """Detect problematic patterns in sequences."""
        issues = {}
        
        # Zero sequences
        issues['zero_seq1'] = np.sum(np.abs(seq1) < self.min_eigenvalue_threshold * 10) / len(seq1)
        issues['zero_seq2'] = np.sum(np.abs(seq2) < self.min_eigenvalue_threshold * 10) / len(seq2)
        issues['both_zero'] = min(issues['zero_seq1'], issues['zero_seq2'])
        
        # Constant sequences
        issues['constant_seq1'] = 1.0 if np.std(seq1) < 1e-10 else 0.0
        issues['constant_seq2'] = 1.0 if np.std(seq2) < 1e-10 else 0.0
        
        # Nearly identical sequences
        if len(seq1) == len(seq2):
            mse = np.mean((seq1 - seq2) ** 2)
            issues['nearly_identical'] = 1.0 if mse < 1e-10 else 0.0
            try:
                correlation = np.corrcoef(seq1, seq2)[0, 1]
                if np.isnan(correlation):
                    correlation = 0.0
            except:
                correlation = 0.0
            issues['correlation'] = correlation
        else:
            issues['nearly_identical'] = 0.0
            issues['correlation'] = 0.0
        
        # Range problems
        issues['small_range1'] = 1.0 if (np.max(seq1) - np.min(seq1)) < 1e-8 else 0.0
        issues['small_range2'] = 1.0 if (np.max(seq2) - np.min(seq2)) < 1e-8 else 0.0
        
        return issues
    
    def _apply_normalized_adjustments(self, normalized_distance: float, issues: Dict[str, float]) -> float:
        """Apply adjustments to normalized distance for problematic sequences."""
        adjusted_distance = normalized_distance
        
        # Handle identical sequences (should have similarity close to 1.0)
        if issues['nearly_identical'] > 0.5:
            return 0.0  # Perfect similarity
        
        # Handle both sequences being constant but different
        if issues['constant_seq1'] > 0.5 and issues['constant_seq2'] > 0.5:
            # Check if they're the same constant value
            if abs(issues['correlation'] - 1.0) < 1e-6:
                return 0.0  # Same constant = perfect similarity
            else:
                return 1.0  # Different constants = maximum distance
        
        # Handle sequences with many zeros (CORRECTED: both having zeros means they're similar)
        min_zero_ratio = min(issues['zero_seq1'], issues['zero_seq2'])
        max_zero_ratio = max(issues['zero_seq1'], issues['zero_seq2'])
        
        # If both sequences have many zeros, they should be more similar
        if min_zero_ratio > 0.5 and max_zero_ratio > 0.7:
            # Both sequences are mostly zeros - reduce distance (increase similarity)
            zero_similarity_bonus = min(0.3, min_zero_ratio * 0.5)
            adjusted_distance = max(0.0, adjusted_distance - zero_similarity_bonus)
        elif max_zero_ratio > 0.8 and min_zero_ratio < 0.1:
            # One sequence is mostly zeros, other is not - increase distance
            zero_difference_penalty = min(0.2, (max_zero_ratio - min_zero_ratio) * 0.3)
            adjusted_distance += zero_difference_penalty
        
        # Handle artificially high correlation with low distance
        if issues['correlation'] > 0.98 and normalized_distance < 1e-6:
            # Moderate penalty for suspicious patterns
            adjusted_distance = max(adjusted_distance, 0.1)
        
        return adjusted_distance
    
    def _validate_sequence_diversity(self, seq1: np.ndarray, seq2: np.ndarray, multivariate: bool) -> Dict[str, Any]:
        """Validate that sequences have sufficient diversity for meaningful DTW comparison."""
        issues = {}
        
        # Flatten sequences for analysis
        seq1_flat = seq1.flatten() if seq1.ndim > 1 else seq1
        seq2_flat = seq2.flatten() if seq2.ndim > 1 else seq2
        
        # Check for identical sequences
        if len(seq1_flat) == len(seq2_flat):
            mse = np.mean((seq1_flat - seq2_flat) ** 2)
            if mse < 1e-12:
                logger.warning("Sequences are nearly identical - DTW distance may be artificially low")
                issues['identical_sequences'] = True
        
        # Check for constant sequences
        seq1_variance = np.var(seq1_flat)
        seq2_variance = np.var(seq2_flat)
        
        if seq1_variance < 1e-10:
            logger.warning(f"Sequence 1 is nearly constant (variance: {seq1_variance:.2e})")
            issues['constant_seq1'] = True
        
        if seq2_variance < 1e-10:
            logger.warning(f"Sequence 2 is nearly constant (variance: {seq2_variance:.2e})")
            issues['constant_seq2'] = True
        
        # Check for excessive padding in multivariate case
        if multivariate and seq1.ndim > 1:
            # Count how many values are at the minimum threshold (likely padding)
            padding_ratio_1 = np.sum(seq1 <= self.min_eigenvalue_threshold * 1.1) / seq1.size
            padding_ratio_2 = np.sum(seq2 <= self.min_eigenvalue_threshold * 1.1) / seq2.size
            
            if padding_ratio_1 > 0.7 or padding_ratio_2 > 0.7:
                logger.warning(f"High padding detected: seq1={padding_ratio_1:.2f}, seq2={padding_ratio_2:.2f}")
                logger.warning("This may cause artificial similarities in multivariate DTW")
                issues['excessive_padding'] = True
        
        # Check for poor dynamic range
        seq1_range = np.max(seq1_flat) - np.min(seq1_flat)
        seq2_range = np.max(seq2_flat) - np.min(seq2_flat)
        
        if seq1_range < 1e-8:
            logger.warning(f"Sequence 1 has very small dynamic range: {seq1_range:.2e}")
            issues['poor_range_seq1'] = True
            
        if seq2_range < 1e-8:
            logger.warning(f"Sequence 2 has very small dynamic range: {seq2_range:.2e}")
            issues['poor_range_seq2'] = True
        
        # Report if sequences look good
        if not issues:
            logger.debug("Sequence diversity validation passed - sequences appear suitable for DTW")
        
        return issues
    
    def _compute_multiple_normalizations(self, seq1: np.ndarray, seq2: np.ndarray, distance: float) -> Dict[str, float]:
        """Compute multiple normalization schemes."""
        normalizations = {}
        
        # Current normalization (max length)
        normalizations['current'] = distance / max(len(seq1), len(seq2))
        
        # Range-aware normalization
        range1 = np.max(seq1) - np.min(seq1)
        range2 = np.max(seq2) - np.min(seq2)
        avg_range = (range1 + range2) / 2
        if avg_range > 1e-12:
            normalizations['range_aware'] = distance / avg_range
        else:
            normalizations['range_aware'] = distance
        
        # Path length normalization (more sensitive)
        normalizations['path_length'] = distance / (len(seq1) * len(seq2))
        
        # Standard deviation normalization
        combined_std = np.std(np.concatenate([seq1, seq2]))
        if combined_std > 1e-12:
            normalizations['std_aware'] = distance / combined_std
        else:
            normalizations['std_aware'] = distance
        
        return normalizations
    
    def _select_best_normalization(self, normalizations: Dict[str, float], issues: Dict[str, float]) -> float:
        """Select the best normalization scheme based on sequence characteristics."""
        
        # If sequences have significant range, use range-aware
        if issues['small_range1'] < 0.5 and issues['small_range2'] < 0.5:
            return normalizations['range_aware']
        
        # If sequences are mostly zeros, use path length (more sensitive)
        if max(issues['zero_seq1'], issues['zero_seq2']) > 0.7:
            return normalizations['path_length']
        
        # If sequences are nearly identical, use standard deviation aware
        if issues['nearly_identical'] > 0.5:
            return normalizations['std_aware']
        
        # Default to current scheme
        return normalizations['current']
    
    def compare_eigenvalue_evolution(self,
                                   evolution1: List[torch.Tensor],
                                   evolution2: List[torch.Tensor],
                                   filtration_params1: Optional[List[float]] = None,
                                   filtration_params2: Optional[List[float]] = None,
                                   eigenvalue_index: Optional[int] = None,
                                   multivariate: bool = False,
                                   use_interpolation: bool = True,
                                   match_all_eigenvalues: bool = True,
                                   interpolation_points: Optional[int] = None) -> Dict[str, Any]:
        """Compare eigenvalue evolution between two filtration sequences.
        
        Args:
            evolution1: First eigenvalue evolution sequence
            evolution2: Second eigenvalue evolution sequence
            filtration_params1: Filtration parameters for first sequence
            filtration_params2: Filtration parameters for second sequence
            eigenvalue_index: Index of eigenvalue to compare (None = all)
            multivariate: Whether to use multivariate DTW
            use_interpolation: Whether to use interpolation for multivariate DTW
            match_all_eigenvalues: Whether to compare all eigenvalues (only with interpolation)
            interpolation_points: Number of interpolation points (auto if None)
            
        Returns:
            Dictionary containing:
            - distance: DTW distance between sequences
            - alignment: Optimal alignment path
            - normalized_distance: Distance normalized by sequence lengths
            - alignment_visualization: Data for plotting alignment
            - interpolation_info: Information about interpolation (if used)
        """
        # Validate inputs
        self._validate_evolution_sequences(evolution1, evolution2)
        
        # Extract eigenvalue sequences
        if multivariate and use_interpolation and match_all_eigenvalues:
            # Use new interpolation-based extraction
            seq1, seq2 = self._extract_multivariate_sequences_interpolated(
                evolution1, evolution2,
                filtration_params1, filtration_params2,
                interpolation_points
            )
            interpolation_used = True
        elif multivariate:
            # Use original multivariate extraction (with padding/truncation)
            seq1, seq2 = self._extract_multivariate_sequences(evolution1, evolution2)
            interpolation_used = False
        else:
            # Univariate extraction
            seq1, seq2 = self._extract_univariate_sequences(
                evolution1, evolution2, eigenvalue_index
            )
            interpolation_used = False
        
        # Validate sequence diversity before DTW computation
        diversity_issues = self._validate_sequence_diversity(seq1, seq2, multivariate)
        
        # Compute DTW distance and alignment
        if multivariate and self.method == 'tslearn':
            distance, alignment = self._compute_multivariate_dtw(seq1, seq2)
        else:
            distance, alignment = self._compute_univariate_dtw(seq1, seq2)
        
        # Enhanced normalization with sequence validation
        normalized_distance = self._compute_enhanced_normalization(
            seq1, seq2, distance
        )
        
        # FIXED: Remove aggressive clamping that destroys sensitivity
        # Only ensure non-negative distances, preserve relative differences
        final_normalized_distance = max(0.0, normalized_distance)
        
        # Diagnostic logging for DTW analysis
        seq_info = f"seq_shapes=({seq1.shape}, {seq2.shape})" if hasattr(seq1, 'shape') else f"seq_lens=({len(seq1)}, {len(seq2)})"
        logger.debug(f"DTW comparison: {seq_info}, raw_distance={distance:.6f}, "
                    f"final_normalized={final_normalized_distance:.6f}, multivariate={multivariate}")
        
        # Prepare alignment visualization data
        alignment_viz = self._prepare_alignment_visualization(
            seq1, seq2, alignment, filtration_params1, filtration_params2
        )
        
        # Prepare interpolation info if used
        interpolation_info = None
        if interpolation_used:
            interpolation_info = {
                'method': 'piecewise_linear',
                'num_features': seq1.shape[1] if seq1.ndim > 1 else 1,
                'num_time_points': len(seq1),
                'used_all_eigenvalues': match_all_eigenvalues
            }
        
        return {
            'distance': float(distance),
            'alignment': alignment,
            'normalized_distance': float(final_normalized_distance),
            'raw_normalized_distance': float(normalized_distance),  # Keep original for debugging
            'alignment_visualization': alignment_viz,
            'sequence1_length': len(seq1),
            'sequence2_length': len(seq2),
            'method': self.method,
            'multivariate': multivariate,
            'interpolation_used': interpolation_used,
            'interpolation_info': interpolation_info
        }
    
    def compare_multiple_evolutions(self,
                                  evolutions: List[List[torch.Tensor]],
                                  filtration_params: Optional[List[List[float]]] = None,
                                  eigenvalue_index: Optional[int] = None,
                                  multivariate: bool = False,
                                  use_interpolation: bool = True,
                                  match_all_eigenvalues: bool = True,
                                  interpolation_points: Optional[int] = None) -> np.ndarray:
        """Compare multiple eigenvalue evolutions pairwise.
        
        Args:
            evolutions: List of eigenvalue evolution sequences
            filtration_params: List of filtration parameter sequences
            eigenvalue_index: Index of eigenvalue to compare (None = all)
            multivariate: Whether to use multivariate DTW
            use_interpolation: Whether to use interpolation for multivariate DTW
            match_all_eigenvalues: Whether to compare all eigenvalues (only with interpolation)
            interpolation_points: Number of interpolation points (auto if None)
            
        Returns:
            Symmetric distance matrix of shape (n_evolutions, n_evolutions)
        """
        n_evolutions = len(evolutions)
        distance_matrix = np.zeros((n_evolutions, n_evolutions))
        
        method_desc = "interpolation-based" if (multivariate and use_interpolation) else "standard"
        logger.info(f"Computing pairwise DTW distances for {n_evolutions} evolutions using {method_desc} method")
        
        for i in range(n_evolutions):
            for j in range(i + 1, n_evolutions):
                params1 = filtration_params[i] if filtration_params else None
                params2 = filtration_params[j] if filtration_params else None
                
                result = self.compare_eigenvalue_evolution(
                    evolutions[i], evolutions[j], params1, params2,
                    eigenvalue_index, multivariate,
                    use_interpolation, match_all_eigenvalues, interpolation_points
                )
                
                distance_matrix[i, j] = result['normalized_distance']
                distance_matrix[j, i] = distance_matrix[i, j]  # Symmetric
        
        return distance_matrix
    
    def _validate_evolution_sequences(self,
                                    evolution1: List[torch.Tensor],
                                    evolution2: List[torch.Tensor]) -> None:
        """Validate eigenvalue evolution sequences."""
        if not evolution1 or not evolution2:
            raise ValidationError("Evolution sequences cannot be empty")
        
        # Check that all elements are tensors
        for i, seq in enumerate(evolution1):
            if not isinstance(seq, torch.Tensor):
                raise ValidationError(f"evolution1[{i}] is not a tensor")
        
        for i, seq in enumerate(evolution2):
            if not isinstance(seq, torch.Tensor):
                raise ValidationError(f"evolution2[{i}] is not a tensor")
    
    def _extract_univariate_sequences(self,
                                    evolution1: List[torch.Tensor],
                                    evolution2: List[torch.Tensor],
                                    eigenvalue_index: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Extract univariate eigenvalue sequences."""
        if eigenvalue_index is None:
            eigenvalue_index = 0  # Default to largest eigenvalue
        
        seq1 = []
        seq2 = []
        
        # Extract specified eigenvalue from each step
        for eigenvals in evolution1:
            if len(eigenvals) > eigenvalue_index:
                val = eigenvals[eigenvalue_index].item()
                seq1.append(max(val, self.min_eigenvalue_threshold))
            else:
                seq1.append(self.min_eigenvalue_threshold)
        
        for eigenvals in evolution2:
            if len(eigenvals) > eigenvalue_index:
                val = eigenvals[eigenvalue_index].item()
                seq2.append(max(val, self.min_eigenvalue_threshold))
            else:
                seq2.append(self.min_eigenvalue_threshold)
        
        return np.array(seq1), np.array(seq2)
    
    def _extract_multivariate_sequences(self,
                                       evolution1: List[torch.Tensor],
                                       evolution2: List[torch.Tensor]) -> Tuple[np.ndarray, np.ndarray]:
        """Extract multivariate eigenvalue sequences with improved padding and validation."""
        try:
            # Find maximum number of eigenvalues across all steps
            max_eigenvals1 = max(len(eigenvals) for eigenvals in evolution1 if len(eigenvals) > 0)
            max_eigenvals2 = max(len(eigenvals) for eigenvals in evolution2 if len(eigenvals) > 0)
            max_eigenvals = max(max_eigenvals1, max_eigenvals2)
            
            if max_eigenvals == 0:
                raise ValidationError("No eigenvalues found in evolution sequences")
            
            # Limit the number of eigenvalues to prevent excessive padding
            # Use top eigenvalues only to focus on dominant spectral behavior
            n_eigenvals_to_use = min(max_eigenvals, 20)  # Limit to top 20 eigenvalues
            
            # Create padded sequences - shape (time_steps, features) for tslearn
            seq1 = np.full((len(evolution1), n_eigenvals_to_use), self.min_eigenvalue_threshold, dtype=np.float64)
            seq2 = np.full((len(evolution2), n_eigenvals_to_use), self.min_eigenvalue_threshold, dtype=np.float64)
            
            # Track how much padding is being used
            total_padded_1 = 0
            total_padded_2 = 0
            
            # Fill in eigenvalue data with proper validation
            for i, eigenvals in enumerate(evolution1):
                if len(eigenvals) > 0:
                    vals = eigenvals.detach().cpu().numpy().astype(np.float64)
                    # Apply threshold and ensure positive values
                    vals = np.maximum(vals, self.min_eigenvalue_threshold)
                    # Sort eigenvalues in descending order for consistency
                    vals = np.sort(vals)[::-1]
                    
                    # Take only the top eigenvalues we need
                    n_to_take = min(len(vals), n_eigenvals_to_use)
                    seq1[i, :n_to_take] = vals[:n_to_take]
                    
                    # Track padding
                    if n_to_take < n_eigenvals_to_use:
                        total_padded_1 += (n_eigenvals_to_use - n_to_take)
                else:
                    total_padded_1 += n_eigenvals_to_use
            
            for i, eigenvals in enumerate(evolution2):
                if len(eigenvals) > 0:
                    vals = eigenvals.detach().cpu().numpy().astype(np.float64)
                    # Apply threshold and ensure positive values
                    vals = np.maximum(vals, self.min_eigenvalue_threshold)
                    # Sort eigenvalues in descending order for consistency
                    vals = np.sort(vals)[::-1]
                    
                    # Take only the top eigenvalues we need
                    n_to_take = min(len(vals), n_eigenvals_to_use)
                    seq2[i, :n_to_take] = vals[:n_to_take]
                    
                    # Track padding
                    if n_to_take < n_eigenvals_to_use:
                        total_padded_2 += (n_eigenvals_to_use - n_to_take)
                else:
                    total_padded_2 += n_eigenvals_to_use
            
            # Calculate padding ratios
            total_elements_1 = seq1.size
            total_elements_2 = seq2.size
            padding_ratio_1 = total_padded_1 / total_elements_1
            padding_ratio_2 = total_padded_2 / total_elements_2
            
            # Warn if too much padding (artificial similarity risk)
            if padding_ratio_1 > 0.5 or padding_ratio_2 > 0.5:
                logger.warning(f"High padding ratios detected: seq1={padding_ratio_1:.2f}, seq2={padding_ratio_2:.2f}")
                logger.warning("This may cause artificial similarities in DTW comparison")
            
            # Validate output shapes
            if seq1.shape[1] != seq2.shape[1]:
                logger.warning(f"Shape mismatch after padding: {seq1.shape} vs {seq2.shape}")
            
            logger.debug(f"Extracted multivariate sequences: {seq1.shape}, {seq2.shape}")
            logger.debug(f"Used {n_eigenvals_to_use}/{max_eigenvals} eigenvalues, padding ratios: {padding_ratio_1:.2f}, {padding_ratio_2:.2f}")
            
            return seq1, seq2
            
        except Exception as e:
            logger.error(f"Failed to extract multivariate sequences: {e}")
            raise ComputationError(f"Multivariate sequence extraction failed: {e}")
    
    def _organize_eigenvalue_sequences(self, evolution: List[torch.Tensor]) -> Dict[str, Any]:
        """Organize eigenvalues into trackable sequences.
        
        Args:
            evolution: List of eigenvalue tensors for each filtration step
            
        Returns:
            Dictionary containing:
            - sequences: List[List[float]], where sequences[i] tracks i-th eigenvalue
            - exists_mask: List[List[bool]], existence mask for each eigenvalue
            - num_eigenvalues_per_step: List[int], number of eigenvalues at each step
            - max_eigenvalues: int, maximum number of eigenvalues across all steps
        """
        # Find maximum number of eigenvalues
        num_eigenvalues_per_step = [len(eigenvals) for eigenvals in evolution]
        max_eigenvalues = max(num_eigenvalues_per_step) if num_eigenvalues_per_step else 0
        
        # Initialize sequences and masks
        sequences = [[] for _ in range(max_eigenvalues)]
        exists_mask = [[] for _ in range(max_eigenvalues)]
        
        # Organize eigenvalues by index
        for step_idx, eigenvals in enumerate(evolution):
            n_eigen = len(eigenvals)
            
            # Sort eigenvalues in ascending order for consistent tracking
            if n_eigen > 0:
                sorted_eigenvals = torch.sort(eigenvals)[0]
                eigenvals_array = sorted_eigenvals.detach().cpu().numpy()
            else:
                eigenvals_array = np.array([])
            
            # Track each eigenvalue
            for eigen_idx in range(max_eigenvalues):
                if eigen_idx < n_eigen:
                    # Eigenvalue exists at this step
                    sequences[eigen_idx].append(float(eigenvals_array[eigen_idx]))
                    exists_mask[eigen_idx].append(True)
                else:
                    # Eigenvalue doesn't exist at this step
                    sequences[eigen_idx].append(None)
                    exists_mask[eigen_idx].append(False)
        
        return {
            'sequences': sequences,
            'exists_mask': exists_mask,
            'num_eigenvalues_per_step': num_eigenvalues_per_step,
            'max_eigenvalues': max_eigenvalues
        }
    
    def _create_normalized_position_mapping(self, 
                                          filtration_params: List[float],
                                          sequence: List[Optional[float]],
                                          exists_mask: List[bool]) -> Tuple[np.ndarray, np.ndarray]:
        """Map eigenvalue sequence to normalized positions.
        
        Args:
            filtration_params: Original filtration parameters
            sequence: Eigenvalue sequence (may contain None values)
            exists_mask: Boolean mask indicating where eigenvalues exist
            
        Returns:
            normalized_positions: Array of normalized positions [0, 1] where eigenvalue exists
            values: Corresponding eigenvalue values
        """
        # Extract valid (position, value) pairs
        valid_positions = []
        valid_values = []
        
        # Normalize filtration parameters to [0, 1]
        params_array = np.array(filtration_params)
        param_min = params_array.min()
        param_max = params_array.max()
        
        # Handle edge case of constant parameters
        if param_max - param_min < 1e-10:
            # Distribute evenly in [0, 1]
            normalized_params = np.linspace(0, 1, len(filtration_params))
        else:
            # Standard normalization
            normalized_params = (params_array - param_min) / (param_max - param_min)
        
        # Extract valid points
        for i, (val, exists) in enumerate(zip(sequence, exists_mask)):
            if exists and val is not None:
                valid_positions.append(normalized_params[i])
                valid_values.append(max(val, self.min_eigenvalue_threshold))
        
        return np.array(valid_positions), np.array(valid_values)
    
    def _interpolate_eigenvalue_sequence(self,
                                       positions: np.ndarray,
                                       values: np.ndarray,
                                       target_positions: np.ndarray) -> np.ndarray:
        """Perform piecewise linear interpolation of eigenvalue sequence.
        
        Args:
            positions: Normalized positions [0,1] where eigenvalues exist
            values: Eigenvalue values at those positions
            target_positions: Target positions for interpolation
            
        Returns:
            Interpolated eigenvalue values at target positions
        """
        from scipy.interpolate import interp1d
        
        # Handle edge cases
        if len(positions) == 0:
            # No valid eigenvalues - return threshold values
            return np.full_like(target_positions, self.min_eigenvalue_threshold)
        
        if len(positions) == 1:
            # Single eigenvalue - constant interpolation
            return np.full_like(target_positions, values[0])
        
        # Create interpolator with constant extrapolation
        interpolator = interp1d(
            positions, values,
            kind='linear',
            bounds_error=False,
            fill_value=(values[0], values[-1])  # Constant extrapolation
        )
        
        # Interpolate at target positions
        interpolated = interpolator(target_positions)
        
        # Ensure all values are above threshold
        interpolated = np.maximum(interpolated, self.min_eigenvalue_threshold)
        
        return interpolated
    
    def _determine_common_sampling_points(self,
                                        evolution1: List[torch.Tensor],
                                        evolution2: List[torch.Tensor],
                                        filtration_params1: List[float],
                                        filtration_params2: List[float],
                                        interpolation_points: Optional[int] = None) -> np.ndarray:
        """Determine optimal sampling points for both sequences.
        
        Args:
            evolution1: First eigenvalue evolution
            evolution2: Second eigenvalue evolution
            filtration_params1: First filtration parameters
            filtration_params2: Second filtration parameters
            interpolation_points: Number of interpolation points (auto if None)
            
        Returns:
            Array of normalized sampling points in [0, 1]
        """
        if interpolation_points is not None:
            # Use fixed number of points
            return np.linspace(0, 1, interpolation_points)
        
        # Normalize both parameter sets
        params1_array = np.array(filtration_params1)
        params2_array = np.array(filtration_params2)
        
        # Normalize to [0, 1]
        def normalize_params(params):
            p_min, p_max = params.min(), params.max()
            if p_max - p_min < 1e-10:
                return np.linspace(0, 1, len(params))
            return (params - p_min) / (p_max - p_min)
        
        norm_params1 = normalize_params(params1_array)
        norm_params2 = normalize_params(params2_array)
        
        # Combine all normalized points and sort
        all_points = np.concatenate([norm_params1, norm_params2])
        unique_points = np.unique(all_points)
        
        # Optionally add intermediate points for better resolution
        if len(unique_points) < 1.5 * max(len(filtration_params1), len(filtration_params2)):
            # Add midpoints between consecutive unique points
            midpoints = []
            for i in range(len(unique_points) - 1):
                midpoints.append((unique_points[i] + unique_points[i+1]) / 2)
            
            if midpoints:
                all_points_with_midpoints = np.concatenate([unique_points, midpoints])
                unique_points = np.unique(all_points_with_midpoints)
        
        return unique_points
    
    def _match_eigenvalue_sequences(self,
                                  organized1: Dict[str, Any],
                                  organized2: Dict[str, Any]) -> List[Tuple[int, int]]:
        """Match eigenvalue sequences between two models.
        
        Args:
            organized1: Organized eigenvalue data from first model
            organized2: Organized eigenvalue data from second model
            
        Returns:
            List of (idx1, idx2) tuples indicating matched eigenvalue indices
        """
        n_eigen1 = organized1['max_eigenvalues']
        n_eigen2 = organized2['max_eigenvalues']
        
        # Direct matching for common eigenvalues (by position in sorted spectrum)
        n_common = min(n_eigen1, n_eigen2)
        matches = [(i, i) for i in range(n_common)]
        
        # Log matching information
        logger.debug(f"Eigenvalue matching: model1 has {n_eigen1} eigenvalues, "
                    f"model2 has {n_eigen2} eigenvalues, {n_common} direct matches")
        
        # For now, we use position-based matching only
        # More sophisticated matching (correlation-based) can be added later if needed
        
        return matches
    
    def _extract_multivariate_sequences_interpolated(self,
                                                   evolution1: List[torch.Tensor],
                                                   evolution2: List[torch.Tensor],
                                                   filtration_params1: Optional[List[float]] = None,
                                                   filtration_params2: Optional[List[float]] = None,
                                                   interpolation_points: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Extract multivariate eigenvalue sequences using interpolation.
        
        This method preserves ALL eigenvalues and uses interpolation to handle
        different numbers of eigenvalues between models.
        
        Args:
            evolution1: First eigenvalue evolution sequence
            evolution2: Second eigenvalue evolution sequence
            filtration_params1: Filtration parameters for first sequence
            filtration_params2: Filtration parameters for second sequence
            interpolation_points: Number of interpolation points (auto if None)
            
        Returns:
            Tuple of interpolated multivariate sequences with shape (n_steps, n_features)
        """
        try:
            # Use default filtration parameters if not provided
            if filtration_params1 is None:
                filtration_params1 = list(range(len(evolution1)))
            if filtration_params2 is None:
                filtration_params2 = list(range(len(evolution2)))
            
            # Organize eigenvalue sequences
            organized1 = self._organize_eigenvalue_sequences(evolution1)
            organized2 = self._organize_eigenvalue_sequences(evolution2)
            
            # Determine common sampling points
            sampling_points = self._determine_common_sampling_points(
                evolution1, evolution2, 
                filtration_params1, filtration_params2,
                interpolation_points
            )
            
            # Match eigenvalue sequences
            matches = self._match_eigenvalue_sequences(organized1, organized2)
            
            # Interpolate matched sequences
            seq1_interpolated = []
            seq2_interpolated = []
            
            for idx1, idx2 in matches:
                # Get sequences and masks
                sequence1 = organized1['sequences'][idx1]
                mask1 = organized1['exists_mask'][idx1]
                
                sequence2 = organized2['sequences'][idx2]
                mask2 = organized2['exists_mask'][idx2]
                
                # Create normalized position mappings
                positions1, values1 = self._create_normalized_position_mapping(
                    filtration_params1, sequence1, mask1
                )
                positions2, values2 = self._create_normalized_position_mapping(
                    filtration_params2, sequence2, mask2
                )
                
                # Interpolate at common sampling points
                if len(positions1) > 0:
                    interp1 = self._interpolate_eigenvalue_sequence(
                        positions1, values1, sampling_points
                    )
                else:
                    # No valid eigenvalues for this index in model 1
                    interp1 = np.full_like(sampling_points, self.min_eigenvalue_threshold)
                
                if len(positions2) > 0:
                    interp2 = self._interpolate_eigenvalue_sequence(
                        positions2, values2, sampling_points
                    )
                else:
                    # No valid eigenvalues for this index in model 2
                    interp2 = np.full_like(sampling_points, self.min_eigenvalue_threshold)
                
                seq1_interpolated.append(interp1)
                seq2_interpolated.append(interp2)
            
            # Convert to numpy arrays with shape (n_steps, n_features)
            if seq1_interpolated:
                seq1 = np.column_stack(seq1_interpolated)
                seq2 = np.column_stack(seq2_interpolated)
            else:
                # No eigenvalues to compare
                seq1 = np.empty((len(sampling_points), 0))
                seq2 = np.empty((len(sampling_points), 0))
            
            logger.info(f"Interpolated sequences: {seq1.shape}, {seq2.shape} "
                       f"({len(matches)} matched eigenvalues, {len(sampling_points)} time points)")
            
            return seq1, seq2
            
        except Exception as e:
            logger.error(f"Failed to extract interpolated multivariate sequences: {e}")
            raise ComputationError(f"Interpolated sequence extraction failed: {e}")
    
    def _compute_univariate_dtw(self, seq1: np.ndarray, seq2: np.ndarray) -> Tuple[float, List[Tuple[int, int]]]:
        """Compute univariate DTW distance and alignment."""
        if self.method == 'dtaidistance':
            return self._dtaidistance_univariate(seq1, seq2)
        elif self.method == 'tslearn':
            return self._tslearn_univariate(seq1, seq2)
        elif self.method == 'dtw-python':
            return self._dtw_python_univariate(seq1, seq2)
        else:
            raise ComputationError(f"Unknown DTW method: {self.method}")
    
    def _compute_multivariate_dtw(self, seq1: np.ndarray, seq2: np.ndarray) -> Tuple[float, List[Tuple[int, int]]]:
        """Compute multivariate DTW distance and alignment."""
        if self.method == 'tslearn':
            return self._tslearn_multivariate(seq1, seq2)
        else:
            # Fallback to univariate DTW on first eigenvalue
            logger.warning(f"Multivariate DTW not available for {self.method}, using univariate")
            return self._compute_univariate_dtw(seq1[:, 0], seq2[:, 0])
    
    def _dtaidistance_univariate(self, seq1: np.ndarray, seq2: np.ndarray) -> Tuple[float, List[Tuple[int, int]]]:
        """Compute DTW using dtaidistance library."""
        try:
            # Convert to float64 for dtaidistance and ensure 1D
            seq1 = np.asarray(seq1, dtype=np.float64).flatten()
            seq2 = np.asarray(seq2, dtype=np.float64).flatten()
            
            # Compute distance first with proper API
            if self.constraint_band > 0:
                window = int(max(len(seq1), len(seq2)) * self.constraint_band)
                # Ensure minimum window size to avoid inf distances
                if window <= 0:
                    # Use unconstrained DTW for very small sequences or small constraint bands
                    distance = dtw.distance(seq1, seq2)
                else:
                    distance = dtw.distance(seq1, seq2, window=window)
            else:
                distance = dtw.distance_fast(seq1, seq2)
            
            # For alignment path, use the simpler API or create diagonal fallback
            try:
                # Try to get warping path - dtaidistance API varies by version
                paths = dtw.warping_paths(seq1, seq2)
                if hasattr(dtw, 'best_path'):
                    alignment = dtw.best_path(paths)
                else:
                    # Fallback: create diagonal alignment
                    min_len = min(len(seq1), len(seq2))
                    alignment = [(i, i) for i in range(min_len)]
            except:
                # Fallback: create diagonal alignment  
                min_len = min(len(seq1), len(seq2))
                alignment = [(i, i) for i in range(min_len)]
            
            return float(distance), alignment
            
        except Exception as e:
            logger.warning(f"dtaidistance computation failed: {e}, using simple euclidean fallback")
            # Fallback to simple euclidean distance
            min_len = min(len(seq1), len(seq2))
            padded_seq1 = np.pad(seq1, (0, max(0, len(seq2) - len(seq1))), mode='constant', constant_values=seq1[-1] if len(seq1) > 0 else 0.0)
            padded_seq2 = np.pad(seq2, (0, max(0, len(seq1) - len(seq2))), mode='constant', constant_values=seq2[-1] if len(seq2) > 0 else 0.0)
            distance = np.linalg.norm(padded_seq1 - padded_seq2)
            alignment = [(i, i) for i in range(min_len)]
            return float(distance), alignment
    
    def _tslearn_univariate(self, seq1: np.ndarray, seq2: np.ndarray) -> Tuple[float, List[Tuple[int, int]]]:
        """Compute DTW using tslearn library."""
        try:
            # Reshape for tslearn (needs 2D arrays)
            seq1_2d = seq1.reshape(-1, 1)
            seq2_2d = seq2.reshape(-1, 1)
            
            # Compute distance and path together for efficiency
            if self.constraint_band > 0:
                global_constraint = "sakoe_chiba"
                sakoe_chiba_radius = int(max(len(seq1), len(seq2)) * self.constraint_band)
                path, distance = dtw_path(seq1_2d, seq2_2d, 
                                        global_constraint=global_constraint,
                                        sakoe_chiba_radius=sakoe_chiba_radius)
            else:
                path, distance = dtw_path(seq1_2d, seq2_2d)
            
            # Convert path to list of tuples
            alignment = [(int(i), int(j)) for i, j in path]
            
            return float(distance), alignment
            
        except Exception as e:
            logger.warning(f"tslearn univariate DTW failed: {e}")
            # Fallback to simple euclidean distance
            min_len = min(len(seq1), len(seq2))
            padded_seq1 = np.pad(seq1, (0, max(0, len(seq2) - len(seq1))), mode='constant', constant_values=seq1[-1])
            padded_seq2 = np.pad(seq2, (0, max(0, len(seq1) - len(seq2))), mode='constant', constant_values=seq2[-1])
            distance = np.linalg.norm(padded_seq1 - padded_seq2)
            alignment = [(i, i) for i in range(min_len)]
            return float(distance), alignment
    
    def _tslearn_multivariate(self, seq1: np.ndarray, seq2: np.ndarray) -> Tuple[float, List[Tuple[int, int]]]:
        """Compute multivariate DTW using tslearn."""
        try:
            # Validate input shapes for multivariate DTW
            if seq1.ndim != 2 or seq2.ndim != 2:
                raise ValidationError(f"Multivariate sequences must be 2D, got shapes {seq1.shape}, {seq2.shape}")
            
            if seq1.shape[1] != seq2.shape[1]:
                raise ValidationError(f"Sequences must have same number of features: {seq1.shape[1]} vs {seq2.shape[1]}")
            
            # Check if log transformation is beneficial
            # Only apply if eigenvalues span multiple orders of magnitude
            combined_seq = np.concatenate([seq1.flatten(), seq2.flatten()])
            valid_values = combined_seq[combined_seq > self.min_eigenvalue_threshold]
            
            use_log_transform = False
            if len(valid_values) > 0:
                value_range = np.max(valid_values) / np.min(valid_values)
                use_log_transform = value_range > 100.0  # Use log if range spans >2 orders of magnitude
            
            if use_log_transform:
                # Apply log transformation for better numerical properties
                seq1_processed = np.log(np.maximum(seq1, self.min_eigenvalue_threshold))
                seq2_processed = np.log(np.maximum(seq2, self.min_eigenvalue_threshold))
                
                # Ensure finite values
                if not (np.isfinite(seq1_processed).all() and np.isfinite(seq2_processed).all()):
                    logger.warning("Non-finite values detected in log-transformed sequences")
                    seq1_processed = np.nan_to_num(seq1_processed, nan=0.0, posinf=10.0, neginf=-10.0)
                    seq2_processed = np.nan_to_num(seq2_processed, nan=0.0, posinf=10.0, neginf=-10.0)
                    
                logger.debug(f"Applied log transformation (range ratio: {value_range:.2f})")
            else:
                # Use original sequences
                seq1_processed = seq1.copy()
                seq2_processed = seq2.copy()
                logger.debug(f"Skipped log transformation (range ratio: {value_range:.2f})")
            
            # Compute DTW with constraints - use combined path computation
            if self.constraint_band > 0:
                global_constraint = "sakoe_chiba"
                sakoe_chiba_radius = int(max(len(seq1_processed), len(seq2_processed)) * self.constraint_band)
                path, distance = dtw_path(seq1_processed, seq2_processed, 
                                        global_constraint=global_constraint,
                                        sakoe_chiba_radius=sakoe_chiba_radius)
            else:
                path, distance = dtw_path(seq1_processed, seq2_processed)
            
            alignment = [(int(i), int(j)) for i, j in path]
            
            return float(distance), alignment
            
        except Exception as e:
            logger.warning(f"tslearn multivariate DTW failed: {e}, falling back to univariate")
            # Fallback to univariate DTW on first dimension
            return self._tslearn_univariate(seq1[:, 0], seq2[:, 0])
    
    def _dtw_python_univariate(self, seq1: np.ndarray, seq2: np.ndarray) -> Tuple[float, List[Tuple[int, int]]]:
        """Compute DTW using dtw-python library."""
        # This is a placeholder - dtw-python has different API
        # For now, fallback to simple distance
        logger.warning("dtw-python implementation not fully supported, using simple distance")
        distance = np.linalg.norm(seq1 - seq2)
        alignment = [(i, i) for i in range(min(len(seq1), len(seq2)))]
        return float(distance), alignment
    
    def _prepare_alignment_visualization(self,
                                       seq1: np.ndarray,
                                       seq2: np.ndarray,
                                       alignment: List[Tuple[int, int]],
                                       filtration_params1: Optional[List[float]] = None,
                                       filtration_params2: Optional[List[float]] = None) -> Dict[str, Any]:
        """Prepare data for alignment visualization."""
        # Default filtration parameters if not provided
        if filtration_params1 is None:
            filtration_params1 = list(range(len(seq1)))
        if filtration_params2 is None:
            filtration_params2 = list(range(len(seq2)))
        
        return {
            'sequence1': seq1.tolist() if seq1.ndim == 1 else seq1.tolist(),
            'sequence2': seq2.tolist() if seq2.ndim == 1 else seq2.tolist(),
            'filtration_params1': filtration_params1,
            'filtration_params2': filtration_params2,
            'alignment': alignment,
            'alignment_quality': len(alignment) / max(len(seq1), len(seq2)) if alignment else 0.0
        }


def create_filtration_dtw_comparator(method: str = 'auto', **kwargs) -> FiltrationDTW:
    """Factory function to create FiltrationDTW instance with validation.
    
    Args:
        method: DTW method to use
        **kwargs: Additional arguments for FiltrationDTW
        
    Returns:
        Configured FiltrationDTW instance
    """
    return FiltrationDTW(method=method, **kwargs)


def quick_dtw_comparison(evolution1: List[torch.Tensor],
                        evolution2: List[torch.Tensor],
                        eigenvalue_index: int = 0) -> float:
    """Quick DTW comparison for single eigenvalue evolution.
    
    Args:
        evolution1: First eigenvalue evolution
        evolution2: Second eigenvalue evolution
        eigenvalue_index: Index of eigenvalue to compare
        
    Returns:
        Normalized DTW distance
    """
    comparator = create_filtration_dtw_comparator()
    result = comparator.compare_eigenvalue_evolution(
        evolution1, evolution2, eigenvalue_index=eigenvalue_index
    )
    return result['normalized_distance']
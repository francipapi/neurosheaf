"""Improved eigenvalue crossing detection and tracking.

This module provides robust detection and tracking of eigenvalue crossings
in persistent spectral analysis, with special handling for near-zero eigenvalues
and support for both normalized and unnormalized Laplacians.

Key Features:
- Adaptive threshold computation for zero detection
- Spectral gap analysis for improved classification
- Crossing detection with subspace angle validation
- Support for normalized Laplacian bounded spectra
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
import logging
from scipy.linalg import subspace_angles

logger = logging.getLogger(__name__)


class EigenvalueCrossingDetector:
    """Detect and track eigenvalue crossings through filtration.
    
    This class provides improved detection of eigenvalue crossings,
    especially for distinguishing true zero eigenvalues from small
    positive eigenvalues that may cross during filtration.
    """
    
    def __init__(self,
                 relative_threshold: float = 1e-12,
                 absolute_threshold: float = 1e-10,
                 crossing_threshold: float = 1e-8,
                 use_spectral_gap: bool = True,
                 gap_ratio_threshold: float = 100.0):
        """Initialize crossing detector.
        
        Args:
            relative_threshold: Relative threshold for zero detection (×max_eigenval)
            absolute_threshold: Absolute threshold for near-zero detection
            crossing_threshold: Threshold for detecting eigenvalue crossings
            use_spectral_gap: Whether to use spectral gap analysis
            gap_ratio_threshold: Ratio threshold for spectral gap detection
        """
        self.relative_threshold = relative_threshold
        self.absolute_threshold = absolute_threshold
        self.crossing_threshold = crossing_threshold
        self.use_spectral_gap = use_spectral_gap
        self.gap_ratio_threshold = gap_ratio_threshold
        
        # Cache for tracking eigenvalue paths
        self._eigenvalue_paths = []
        self._crossing_events = []
        
    def detect_crossings(self,
                        prev_eigenvals: torch.Tensor,
                        curr_eigenvals: torch.Tensor,
                        prev_eigenvecs: torch.Tensor,
                        curr_eigenvecs: torch.Tensor,
                        step: int,
                        is_normalized: bool = False) -> Dict[str, Any]:
        """Detect eigenvalue crossings between consecutive filtration steps.
        
        Args:
            prev_eigenvals: Previous step eigenvalues
            curr_eigenvals: Current step eigenvalues
            prev_eigenvecs: Previous step eigenvectors
            curr_eigenvecs: Current step eigenvectors
            step: Current filtration step index
            is_normalized: Whether using normalized Laplacian
            
        Returns:
            Dictionary with crossing information and classifications
        """
        # Classify eigenvalues at both steps
        prev_classification = self.classify_eigenvalues(
            prev_eigenvals, is_normalized
        )
        curr_classification = self.classify_eigenvalues(
            curr_eigenvals, is_normalized
        )
        
        # Match eigenspaces using subspace angles
        matches = self._match_eigenspaces(
            prev_eigenvals, curr_eigenvals,
            prev_eigenvecs, curr_eigenvecs
        )
        
        # Detect crossings
        crossings = []
        for i, j, similarity in matches:
            if i < len(prev_eigenvals) - 1 and j < len(curr_eigenvals) - 1:
                # Check if eigenvalue order changed
                if self._check_order_change(i, j, prev_eigenvals, curr_eigenvals):
                    crossing = self._create_crossing_event(
                        i, j, prev_eigenvals, curr_eigenvals,
                        prev_classification, curr_classification,
                        similarity, step
                    )
                    crossings.append(crossing)
        
        # Detect near-zero transitions
        transitions = self._detect_zero_transitions(
            prev_eigenvals, curr_eigenvals,
            prev_classification, curr_classification,
            matches, step
        )
        
        result = {
            'crossings': crossings,
            'transitions': transitions,
            'prev_classification': prev_classification,
            'curr_classification': curr_classification,
            'matches': matches,
            'step': step
        }
        
        # Update tracking
        self._crossing_events.extend(crossings)
        
        logger.debug(f"Step {step}: {len(crossings)} crossings, "
                    f"{len(transitions)} zero transitions detected")
        
        return result
    
    def classify_eigenvalues(self,
                            eigenvals: torch.Tensor,
                            is_normalized: bool = False) -> Dict[str, Any]:
        """Classify eigenvalues with improved thresholds.
        
        Args:
            eigenvals: Array of eigenvalues to classify
            is_normalized: Whether this is a normalized Laplacian
            
        Returns:
            Classification dictionary with categories and thresholds
        """
        if len(eigenvals) == 0:
            return {
                'numerical_zeros': torch.zeros(0, dtype=torch.bool),
                'true_zeros': torch.zeros(0, dtype=torch.bool),
                'small_positive': torch.zeros(0, dtype=torch.bool),
                'positive_spectrum': torch.zeros(0, dtype=torch.bool),
                'threshold': self.absolute_threshold
            }
        
        # Compute adaptive threshold
        max_eigenval = torch.max(torch.abs(eigenvals))
        
        # Adjust for normalized Laplacian
        if is_normalized:
            rel_threshold = min(self.relative_threshold, 1e-10)
            abs_threshold = min(self.absolute_threshold, 1e-12)
        else:
            rel_threshold = self.relative_threshold
            abs_threshold = self.absolute_threshold
        
        # Combined threshold
        rel_cutoff = rel_threshold * max_eigenval
        effective_threshold = max(rel_cutoff, abs_threshold)
        
        # Spectral gap detection
        if self.use_spectral_gap and len(eigenvals) > 1:
            gap_threshold = self._detect_spectral_gap(eigenvals)
            if gap_threshold is not None:
                effective_threshold = max(effective_threshold, gap_threshold)
        
        # Classification
        numerical_zeros = eigenvals < -effective_threshold
        true_zeros = (eigenvals >= -effective_threshold) & (eigenvals < effective_threshold)
        small_positive = (eigenvals >= effective_threshold) & (eigenvals < 10 * effective_threshold)
        positive_spectrum = eigenvals >= 10 * effective_threshold
        
        return {
            'numerical_zeros': numerical_zeros,
            'true_zeros': true_zeros,
            'small_positive': small_positive,
            'positive_spectrum': positive_spectrum,
            'threshold': effective_threshold,
            'is_normalized': is_normalized
        }
    
    def _detect_spectral_gap(self, eigenvals: torch.Tensor) -> Optional[float]:
        """Detect spectral gap for improved zero classification.
        
        Args:
            eigenvals: Sorted eigenvalues
            
        Returns:
            Gap threshold if significant gap found, None otherwise
        """
        sorted_eigenvals = torch.sort(eigenvals)[0]
        
        # Look for gap in first few eigenvalues
        for i in range(min(10, len(sorted_eigenvals) - 1)):
            if sorted_eigenvals[i] >= 0 and sorted_eigenvals[i+1] > 0:
                gap_ratio = sorted_eigenvals[i+1] / (sorted_eigenvals[i] + 1e-16)
                
                if gap_ratio > self.gap_ratio_threshold:
                    # Significant gap found
                    gap_threshold = (sorted_eigenvals[i] + sorted_eigenvals[i+1]) / 2
                    logger.debug(f"Spectral gap detected: λ_{i}={sorted_eigenvals[i]:.2e}, "
                               f"λ_{i+1}={sorted_eigenvals[i+1]:.2e}, "
                               f"gap_threshold={gap_threshold:.2e}")
                    return gap_threshold.item()
        
        return None
    
    def _match_eigenspaces(self,
                          prev_eigenvals: torch.Tensor,
                          curr_eigenvals: torch.Tensor,
                          prev_eigenvecs: torch.Tensor,
                          curr_eigenvecs: torch.Tensor) -> List[Tuple[int, int, float]]:
        """Match eigenspaces using subspace angles.
        
        Args:
            prev_eigenvals: Previous eigenvalues
            curr_eigenvals: Current eigenvalues
            prev_eigenvecs: Previous eigenvectors
            curr_eigenvecs: Current eigenvectors
            
        Returns:
            List of (prev_idx, curr_idx, similarity) matches
        """
        matches = []
        n_prev = len(prev_eigenvals)
        n_curr = len(curr_eigenvals)
        
        # Compute similarity matrix using subspace angles
        similarity_matrix = torch.zeros(n_prev, n_curr)
        
        for i in range(n_prev):
            for j in range(n_curr):
                # Use cosine similarity for single eigenvectors
                v_prev = prev_eigenvecs[:, i:i+1]
                v_curr = curr_eigenvecs[:, j:j+1]
                
                # Compute angle between eigenvectors
                cos_angle = torch.abs(torch.dot(v_prev.squeeze(), v_curr.squeeze()))
                similarity_matrix[i, j] = cos_angle
        
        # Greedy matching
        used_prev = set()
        used_curr = set()
        
        # Sort by similarity
        similarities = []
        for i in range(n_prev):
            for j in range(n_curr):
                similarities.append((similarity_matrix[i, j].item(), i, j))
        
        similarities.sort(reverse=True)
        
        for sim, i, j in similarities:
            if i not in used_prev and j not in used_curr and sim > 0.5:
                matches.append((i, j, sim))
                used_prev.add(i)
                used_curr.add(j)
        
        return matches
    
    def _check_order_change(self,
                           prev_idx: int,
                           curr_idx: int,
                           prev_eigenvals: torch.Tensor,
                           curr_eigenvals: torch.Tensor) -> bool:
        """Check if eigenvalue order changed (crossing occurred).
        
        Args:
            prev_idx: Previous eigenvalue index
            curr_idx: Current eigenvalue index
            prev_eigenvals: Previous eigenvalues
            curr_eigenvals: Current eigenvalues
            
        Returns:
            True if crossing detected
        """
        # Check if relative order changed
        if prev_idx > 0:
            # Was this eigenvalue smaller than the previous one?
            was_smaller = prev_eigenvals[prev_idx] < prev_eigenvals[prev_idx - 1]
            
            if curr_idx > 0:
                is_smaller = curr_eigenvals[curr_idx] < curr_eigenvals[curr_idx - 1]
                if was_smaller != is_smaller:
                    return True
        
        if prev_idx < len(prev_eigenvals) - 1:
            # Was this eigenvalue larger than the next one?
            was_larger = prev_eigenvals[prev_idx] > prev_eigenvals[prev_idx + 1]
            
            if curr_idx < len(curr_eigenvals) - 1:
                is_larger = curr_eigenvals[curr_idx] > curr_eigenvals[curr_idx + 1]
                if was_larger != is_larger:
                    return True
        
        return False
    
    def _create_crossing_event(self,
                              prev_idx: int,
                              curr_idx: int,
                              prev_eigenvals: torch.Tensor,
                              curr_eigenvals: torch.Tensor,
                              prev_classification: Dict,
                              curr_classification: Dict,
                              similarity: float,
                              step: int) -> Dict[str, Any]:
        """Create crossing event dictionary.
        
        Args:
            prev_idx: Previous eigenvalue index
            curr_idx: Current eigenvalue index
            prev_eigenvals: Previous eigenvalues
            curr_eigenvals: Current eigenvalues
            prev_classification: Previous classification
            curr_classification: Current classification
            similarity: Eigenspace similarity
            step: Filtration step
            
        Returns:
            Crossing event dictionary
        """
        prev_val = prev_eigenvals[prev_idx].item()
        curr_val = curr_eigenvals[curr_idx].item()
        
        # Determine crossing type
        prev_category = self._get_category(prev_idx, prev_classification)
        curr_category = self._get_category(curr_idx, curr_classification)
        
        crossing_type = 'standard'
        if prev_category == 'true_zeros' or curr_category == 'true_zeros':
            crossing_type = 'near_zero'
        elif prev_category == 'small_positive' or curr_category == 'small_positive':
            crossing_type = 'small_eigenvalue'
        
        return {
            'step': step,
            'prev_idx': prev_idx,
            'curr_idx': curr_idx,
            'prev_value': prev_val,
            'curr_value': curr_val,
            'value_change': curr_val - prev_val,
            'prev_category': prev_category,
            'curr_category': curr_category,
            'crossing_type': crossing_type,
            'eigenspace_similarity': similarity
        }
    
    def _get_category(self, idx: int, classification: Dict) -> str:
        """Get category for eigenvalue at index."""
        if classification['numerical_zeros'][idx]:
            return 'numerical_zeros'
        elif classification['true_zeros'][idx]:
            return 'true_zeros'
        elif classification['small_positive'][idx]:
            return 'small_positive'
        else:
            return 'positive_spectrum'
    
    def _detect_zero_transitions(self,
                                prev_eigenvals: torch.Tensor,
                                curr_eigenvals: torch.Tensor,
                                prev_classification: Dict,
                                curr_classification: Dict,
                                matches: List[Tuple[int, int, float]],
                                step: int) -> List[Dict[str, Any]]:
        """Detect transitions to/from zero eigenvalues.
        
        Args:
            prev_eigenvals: Previous eigenvalues
            curr_eigenvals: Current eigenvalues
            prev_classification: Previous classification
            curr_classification: Current classification
            matches: Eigenspace matches
            step: Filtration step
            
        Returns:
            List of zero transition events
        """
        transitions = []
        
        for prev_idx, curr_idx, similarity in matches:
            prev_is_zero = prev_classification['true_zeros'][prev_idx]
            curr_is_zero = curr_classification['true_zeros'][curr_idx]
            
            if prev_is_zero and not curr_is_zero:
                # Zero to non-zero transition
                transitions.append({
                    'type': 'zero_to_nonzero',
                    'prev_idx': prev_idx,
                    'curr_idx': curr_idx,
                    'prev_value': prev_eigenvals[prev_idx].item(),
                    'curr_value': curr_eigenvals[curr_idx].item(),
                    'step': step,
                    'similarity': similarity
                })
            elif not prev_is_zero and curr_is_zero:
                # Non-zero to zero transition
                transitions.append({
                    'type': 'nonzero_to_zero',
                    'prev_idx': prev_idx,
                    'curr_idx': curr_idx,
                    'prev_value': prev_eigenvals[prev_idx].item(),
                    'curr_value': curr_eigenvals[curr_idx].item(),
                    'step': step,
                    'similarity': similarity
                })
        
        return transitions
    
    def get_crossing_summary(self) -> Dict[str, Any]:
        """Get summary of all detected crossings.
        
        Returns:
            Summary dictionary with crossing statistics
        """
        if not self._crossing_events:
            return {
                'total_crossings': 0,
                'near_zero_crossings': 0,
                'small_eigenvalue_crossings': 0,
                'standard_crossings': 0
            }
        
        near_zero = sum(1 for c in self._crossing_events if c['crossing_type'] == 'near_zero')
        small_eigenvalue = sum(1 for c in self._crossing_events if c['crossing_type'] == 'small_eigenvalue')
        standard = sum(1 for c in self._crossing_events if c['crossing_type'] == 'standard')
        
        return {
            'total_crossings': len(self._crossing_events),
            'near_zero_crossings': near_zero,
            'small_eigenvalue_crossings': small_eigenvalue,
            'standard_crossings': standard,
            'crossing_events': self._crossing_events
        }
#!/usr/bin/env python3
"""
Test Our DTW Wrapper Implementation
"""

import numpy as np
import sys
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from neurosheaf.utils.dtw_similarity import FiltrationDTW


def test_our_dtw_wrapper():
    """Test our DTW wrapper with the same sequences that work in the library."""
    print("🧪 Testing Our DTW Wrapper")
    print("=" * 50)
    
    dtw_comparator = FiltrationDTW()
    
    # Test 1: Different sequences
    seq1 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    seq2 = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
    
    print(f"Test 1 - Different sequences:")
    print(f"  seq1: {seq1}")
    print(f"  seq2: {seq2}")
    
    try:
        distance, alignment = dtw_comparator._compute_univariate_dtw(seq1, seq2)
        normalized = dtw_comparator._compute_enhanced_normalization(seq1, seq2, distance)
        print(f"  Raw distance: {distance}")
        print(f"  Normalized distance: {normalized}")
    except Exception as e:
        print(f"  ❌ Our DTW failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 2: Identical sequences
    seq3 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    seq4 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    
    print(f"\nTest 2 - Identical sequences:")
    print(f"  seq3: {seq3}")
    print(f"  seq4: {seq4}")
    
    try:
        distance2, alignment2 = dtw_comparator._compute_univariate_dtw(seq3, seq4)
        normalized2 = dtw_comparator._compute_enhanced_normalization(seq3, seq4, distance2)
        print(f"  Raw distance: {distance2}")
        print(f"  Normalized distance: {normalized2}")
    except Exception as e:
        print(f"  ❌ Our DTW failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 3: Model-like sequences (identical)
    seq5 = np.array([0.341550, 0.341550, 0.341550, 0.341550, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000])
    seq6 = np.array([0.341550, 0.341550, 0.341550, 0.341550, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000])
    
    print(f"\nTest 3 - Model-like sequences (identical):")
    print(f"  seq5: {seq5}")
    print(f"  seq6: {seq6}")
    
    try:
        distance3, alignment3 = dtw_comparator._compute_univariate_dtw(seq5, seq6)
        normalized3 = dtw_comparator._compute_enhanced_normalization(seq5, seq6, distance3)
        print(f"  Raw distance: {distance3}")
        print(f"  Normalized distance: {normalized3}")
    except Exception as e:
        print(f"  ❌ Our DTW failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 4: Check what happens in the normalization step
    print(f"\nTest 4 - Debug normalization step:")
    
    distance_input = 0.0  # Known good distance
    try:
        # Test normalization directly
        issues = dtw_comparator._detect_sequence_issues(seq5, seq6)
        print(f"  Issues detected: {issues}")
        
        normalizations = dtw_comparator._compute_multiple_normalizations(seq5, seq6, distance_input)
        print(f"  Normalizations: {normalizations}")
        
        adjusted = dtw_comparator._apply_normalized_adjustments(0.0, issues)
        print(f"  Adjusted distance: {adjusted}")
        
    except Exception as e:
        print(f"  ❌ Normalization failed: {e}")
        import traceback
        traceback.print_exc()


def test_dtaidistance_direct_from_wrapper():
    """Test dtaidistance method directly from our wrapper."""
    print(f"\n🔧 Testing dtaidistance method directly:")
    
    dtw_comparator = FiltrationDTW()
    
    # Test sequences that we know work
    seq1 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    seq2 = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
    
    try:
        distance, alignment = dtw_comparator._dtaidistance_univariate(seq1, seq2)
        print(f"  Different sequences - Raw distance: {distance}")
        print(f"  Alignment length: {len(alignment)}")
    except Exception as e:
        print(f"  ❌ dtaidistance method failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test identical sequences
    seq3 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    seq4 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    
    try:
        distance2, alignment2 = dtw_comparator._dtaidistance_univariate(seq3, seq4)
        print(f"  Identical sequences - Raw distance: {distance2}")
        print(f"  Alignment length: {len(alignment2)}")
    except Exception as e:
        print(f"  ❌ dtaidistance method failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("🚀 Testing Our DTW Wrapper Implementation")
    print("=" * 60)
    
    test_our_dtw_wrapper()
    test_dtaidistance_direct_from_wrapper()
    
    print(f"\n📋 Summary:")
    print("- Identifying where our DTW wrapper fails")
    print("- Testing normalization step separately")
    print("- Finding the source of inf distances")
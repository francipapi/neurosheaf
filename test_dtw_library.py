#!/usr/bin/env python3
"""
Test DTW Library Directly
"""

import numpy as np

try:
    from dtaidistance import dtw
    DTW_AVAILABLE = True
    print("✅ dtaidistance available")
except ImportError:
    DTW_AVAILABLE = False
    print("❌ dtaidistance not available")

def test_dtaidistance_direct():
    """Test dtaidistance library directly."""
    if not DTW_AVAILABLE:
        print("Cannot test - dtaidistance not available")
        return
    
    print("\n🧪 Testing dtaidistance directly:")
    
    # Test 1: Simple sequences
    seq1 = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
    seq2 = np.array([5.0, 4.0, 3.0, 2.0, 1.0], dtype=np.float64)
    
    print(f"Test 1 - Different sequences:")
    print(f"  seq1: {seq1}")
    print(f"  seq2: {seq2}")
    
    try:
        dist1 = dtw.distance(seq1, seq2)
        print(f"  DTW distance: {dist1}")
    except Exception as e:
        print(f"  ❌ DTW failed: {e}")
    
    try:
        dist1_fast = dtw.distance_fast(seq1, seq2)
        print(f"  DTW fast distance: {dist1_fast}")
    except Exception as e:
        print(f"  ❌ DTW fast failed: {e}")
    
    # Test 2: Identical sequences
    seq3 = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
    seq4 = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
    
    print(f"\nTest 2 - Identical sequences:")
    print(f"  seq3: {seq3}")
    print(f"  seq4: {seq4}")
    
    try:
        dist2 = dtw.distance(seq3, seq4)
        print(f"  DTW distance: {dist2}")
    except Exception as e:
        print(f"  ❌ DTW failed: {e}")
    
    # Test 3: Constant sequences
    seq5 = np.array([1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float64)
    seq6 = np.array([2.0, 2.0, 2.0, 2.0, 2.0], dtype=np.float64)
    
    print(f"\nTest 3 - Constant sequences:")
    print(f"  seq5: {seq5}")
    print(f"  seq6: {seq6}")
    
    try:
        dist3 = dtw.distance(seq5, seq6)
        print(f"  DTW distance: {dist3}")
    except Exception as e:
        print(f"  ❌ DTW failed: {e}")
    
    # Test 4: Zero sequences
    seq7 = np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
    seq8 = np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
    
    print(f"\nTest 4 - Zero sequences:")
    print(f"  seq7: {seq7}")
    print(f"  seq8: {seq8}")
    
    try:
        dist4 = dtw.distance(seq7, seq8)
        print(f"  DTW distance: {dist4}")
    except Exception as e:
        print(f"  ❌ DTW failed: {e}")
    
    # Test 5: Sequences with small values (like our eigenvalues)
    seq9 = np.array([0.341550, 0.341550, 0.341550, 0.341550, 0.000000], dtype=np.float64)
    seq10 = np.array([0.341550, 0.341550, 0.341550, 0.341550, 0.000000], dtype=np.float64)
    
    print(f"\nTest 5 - Eigenvalue-like sequences (identical):")
    print(f"  seq9: {seq9}")
    print(f"  seq10: {seq10}")
    
    try:
        dist5 = dtw.distance(seq9, seq10)
        print(f"  DTW distance: {dist5}")
    except Exception as e:
        print(f"  ❌ DTW failed: {e}")
    
    # Test 6: Sequences with small differences
    seq11 = np.array([0.341550, 0.341550, 0.341550, 0.341550, 0.000000], dtype=np.float64)
    seq12 = np.array([0.341551, 0.341551, 0.341551, 0.341551, 0.000001], dtype=np.float64)
    
    print(f"\nTest 6 - Eigenvalue-like sequences (small difference):")
    print(f"  seq11: {seq11}")
    print(f"  seq12: {seq12}")
    
    try:
        dist6 = dtw.distance(seq11, seq12)
        print(f"  DTW distance: {dist6}")
    except Exception as e:
        print(f"  ❌ DTW failed: {e}")


def test_problematic_model_sequences():
    """Test DTW on the exact sequences from our models."""
    print(f"\n🔍 Testing exact model sequences:")
    
    # These are the exact sequences from our debug output
    seq1_str = ['0.341550', '0.341550', '0.341550', '0.341550', '0.000000', '0.000000', '0.000000', '0.000000', '0.000000', '0.000000']
    seq2_str = ['0.341550', '0.341550', '0.341550', '0.341550', '0.000000', '0.000000', '0.000000', '0.000000', '0.000000', '0.000000']
    
    seq1 = np.array([float(x) for x in seq1_str], dtype=np.float64)
    seq2 = np.array([float(x) for x in seq2_str], dtype=np.float64)
    
    print(f"Model sequences:")
    print(f"  seq1: {seq1}")
    print(f"  seq2: {seq2}")
    print(f"  Identical: {np.allclose(seq1, seq2)}")
    
    if DTW_AVAILABLE:
        try:
            dist = dtw.distance(seq1, seq2)
            print(f"  DTW distance: {dist}")
        except Exception as e:
            print(f"  ❌ DTW failed: {e}")
    
    # Test with manual small difference
    seq2_modified = seq2.copy()
    seq2_modified[0] = 0.341560  # Small change
    
    print(f"\nWith small modification:")
    print(f"  seq1: {seq1}")
    print(f"  seq2_mod: {seq2_modified}")
    
    if DTW_AVAILABLE:
        try:
            dist_mod = dtw.distance(seq1, seq2_modified)
            print(f"  DTW distance: {dist_mod}")
        except Exception as e:
            print(f"  ❌ DTW failed: {e}")


if __name__ == "__main__":
    print("🚀 DTW Library Testing")
    print("=" * 50)
    
    test_dtaidistance_direct()
    test_problematic_model_sequences()
    
    print(f"\n📋 Summary:")
    print("- Testing if DTW library works correctly")
    print("- Testing various sequence types")
    print("- Identifying why our DTW returns inf")
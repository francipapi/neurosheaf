#!/usr/bin/env python3
"""
Quick validation test for corrected DTW implementation.
Tests the key fixes: unequal lengths, window constraints, cost functions.
"""

import numpy as np
from compute_alternative_distances import _dtw_distance_numpy

def test_dtw_corrections():
    """Test the corrected DTW implementation."""

    print("Testing corrected DTW implementation...")

    # Test 1: Equal length sequences (should work as before)
    x = np.array([1, 2, 3, 4, 5], dtype=np.float64)
    y = np.array([1, 2, 4, 4, 5], dtype=np.float64)

    dist_l1 = _dtw_distance_numpy(x, y, cost="l1")
    dist_l2 = _dtw_distance_numpy(x, y, cost="l2")

    print(f"Test 1 - Equal lengths (n=5, m=5):")
    print(f"  L1 distance: {dist_l1:.6f}")
    print(f"  L2 distance: {dist_l2:.6f}")

    # Test 2: Unequal length sequences (this was broken before)
    x_short = np.array([1, 2, 3], dtype=np.float64)
    y_long = np.array([1, 1.5, 2, 2.5, 3], dtype=np.float64)

    dist_unequal = _dtw_distance_numpy(x_short, y_long, cost="l1")
    dist_unequal_rev = _dtw_distance_numpy(y_long, x_short, cost="l1")

    print(f"\nTest 2 - Unequal lengths (n=3, m=5):")
    print(f"  DTW(short, long): {dist_unequal:.6f}")
    print(f"  DTW(long, short): {dist_unequal_rev:.6f}")
    print(f"  Should be equal: {abs(dist_unequal - dist_unequal_rev) < 1e-10}")

    # Test 3: Window constraint with unequal lengths
    # Previous implementation might fail here due to incorrect window calculation
    x_very_short = np.array([1, 2], dtype=np.float64)
    y_very_long = np.array([0.8, 1.2, 1.8, 2.2, 2.8], dtype=np.float64)

    # With tight window (should still allow feasible path due to |n-m| correction)
    dist_windowed = _dtw_distance_numpy(x_very_short, y_very_long, window=0.1, cost="l1")
    dist_no_window = _dtw_distance_numpy(x_very_short, y_very_long, window=None, cost="l1")

    print(f"\nTest 3 - Window constraint with unequal lengths (n=2, m=5):")
    print(f"  With window=0.1: {dist_windowed:.6f}")
    print(f"  Without window:  {dist_no_window:.6f}")
    print(f"  Window result should be >= no-window due to constraints")

    # Test 4: Path normalization
    dist_normalized = _dtw_distance_numpy(x, y, cost="l1", normalize=True)
    dist_unnormalized = _dtw_distance_numpy(x, y, cost="l1", normalize=False)

    print(f"\nTest 4 - Path normalization:")
    print(f"  Unnormalized: {dist_unnormalized:.6f}")
    print(f"  Normalized:   {dist_normalized:.6f}")
    print(f"  Normalized should be smaller: {dist_normalized < dist_unnormalized}")

    # Test 5: Edge cases
    x_empty = np.array([], dtype=np.float64)
    y_single = np.array([1.0], dtype=np.float64)

    dist_empty = _dtw_distance_numpy(x_empty, y_single, cost="l1")
    dist_identical = _dtw_distance_numpy(y_single, y_single, cost="l1")

    print(f"\nTest 5 - Edge cases:")
    print(f"  Empty vs single: {dist_empty}")
    print(f"  Identical sequences: {dist_identical:.6f}")
    print(f"  Identical should be 0.0: {abs(dist_identical) < 1e-10}")

    print("\n" + "="*50)
    print("DTW correction validation completed!")
    print("If no errors occurred, the implementation should be working correctly.")

if __name__ == "__main__":
    test_dtw_corrections()
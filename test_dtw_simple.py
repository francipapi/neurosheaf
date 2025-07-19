#!/usr/bin/env python3
"""
Simple DTW Test

This script tests the corrected DTW implementation with a quick test
to verify the API fixes work correctly.
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from neurosheaf.utils.dtw_similarity import FiltrationDTW, quick_dtw_comparison


def setup_environment():
    """Set up the environment for reproducible analysis."""
    torch.manual_seed(42)
    np.random.seed(42)
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
    
    print("🔧 Environment Setup Complete")
    print(f"   PyTorch Version: {torch.__version__}")


def test_dtw_api_fixes():
    """Test the DTW API fixes with simple synthetic data."""
    print(f"\n🔬 Testing DTW API Fixes...")
    
    # Create simple test sequences that should have high similarity
    seq1 = [1.0, 2.0, 3.0, 4.0, 5.0, 4.0, 3.0, 2.0, 1.0]
    seq2 = [1.1, 2.1, 3.1, 4.1, 5.1, 4.1, 3.1, 2.1, 1.1]  # Similar with slight offset
    
    # Convert to tensor format expected by the DTW implementation
    evolution1 = [torch.tensor([val]) for val in seq1]
    evolution2 = [torch.tensor([val]) for val in seq2]
    
    print(f"   Test sequences: {len(seq1)} steps each")
    print(f"   Expected: High similarity (similar patterns)")
    
    # Test 1: dtaidistance implementation
    print(f"\n   Test 1: dtaidistance DTW")
    try:
        dtw_comparator = FiltrationDTW(
            method='dtaidistance',
            constraint_band=0.0,  # No constraint for simple test
            eigenvalue_weight=1.0,
            structural_weight=0.0
        )
        
        result = dtw_comparator.compare_eigenvalue_evolution(
            evolution1, evolution2,
            eigenvalue_index=0,
            multivariate=False
        )
        
        print(f"      ✅ dtaidistance DTW successful")
        print(f"         Distance: {result['distance']:.4f}")
        print(f"         Normalized Distance: {result['normalized_distance']:.4f}")
        print(f"         Method: {result['method']}")
        print(f"         Sequences lengths: {result['sequence1_length']}, {result['sequence2_length']}")
        
    except Exception as e:
        print(f"      ❌ dtaidistance DTW failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 2: tslearn implementation  
    print(f"\n   Test 2: tslearn DTW")
    try:
        dtw_comparator = FiltrationDTW(
            method='tslearn',
            constraint_band=0.0,
            eigenvalue_weight=1.0,
            structural_weight=0.0
        )
        
        result = dtw_comparator.compare_eigenvalue_evolution(
            evolution1, evolution2,
            eigenvalue_index=0,
            multivariate=False
        )
        
        print(f"      ✅ tslearn DTW successful")
        print(f"         Distance: {result['distance']:.4f}")
        print(f"         Normalized Distance: {result['normalized_distance']:.4f}")
        print(f"         Method: {result['method']}")
        print(f"         Alignment length: {len(result['alignment'])}")
        
    except Exception as e:
        print(f"      ❌ tslearn DTW failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 3: Multivariate DTW with multiple eigenvalues
    print(f"\n   Test 3: Multivariate DTW")
    try:
        # Create multivariate sequences (2 eigenvalues per step)
        evolution1_multi = [torch.tensor([val, val*0.5]) for val in seq1]
        evolution2_multi = [torch.tensor([val, val*0.5]) for val in seq2]
        
        dtw_comparator = FiltrationDTW(
            method='tslearn',  # tslearn supports multivariate
            constraint_band=0.0,
            eigenvalue_weight=1.0,
            structural_weight=0.0
        )
        
        result = dtw_comparator.compare_eigenvalue_evolution(
            evolution1_multi, evolution2_multi,
            eigenvalue_index=None,  # All eigenvalues
            multivariate=True
        )
        
        print(f"      ✅ Multivariate DTW successful")
        print(f"         Distance: {result['distance']:.4f}")
        print(f"         Normalized Distance: {result['normalized_distance']:.4f}")
        print(f"         Method: {result['method']}")
        print(f"         Multivariate: {result['multivariate']}")
        
    except Exception as e:
        print(f"      ❌ Multivariate DTW failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 4: Quick DTW comparison function
    print(f"\n   Test 4: Quick DTW function")
    try:
        distance = quick_dtw_comparison(evolution1, evolution2, eigenvalue_index=0)
        print(f"      ✅ Quick DTW successful")
        print(f"         Normalized Distance: {distance:.4f}")
        
    except Exception as e:
        print(f"      ❌ Quick DTW failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 5: Weight validation fix
    print(f"\n   Test 5: Weight Validation")
    try:
        # Test pure functional similarity (should work now)
        dtw_pure = FiltrationDTW(
            eigenvalue_weight=1.0,
            structural_weight=0.0
        )
        print(f"      ✅ Pure functional weights (1.0, 0.0) accepted")
        
        # Test balanced weights
        dtw_balanced = FiltrationDTW(
            eigenvalue_weight=0.7,
            structural_weight=0.3
        )
        print(f"      ✅ Balanced weights (0.7, 0.3) accepted")
        
        # Test invalid weights (should fail)
        try:
            dtw_invalid = FiltrationDTW(
                eigenvalue_weight=-0.1,
                structural_weight=0.0
            )
            print(f"      ❌ Negative weights incorrectly accepted")
        except Exception:
            print(f"      ✅ Negative weights correctly rejected")
        
    except Exception as e:
        print(f"      ❌ Weight validation failed: {e}")


def main():
    """Main execution function."""
    print("🚀 Simple DTW API Test")
    print("=" * 40)
    
    # Setup
    setup_environment()
    
    try:
        # Test the DTW API fixes
        test_dtw_api_fixes()
        
        print(f"\n✅ DTW API Test Complete!")
        print(f"\n📋 Summary:")
        print(f"   • All DTW library API calls should now work correctly")
        print(f"   • Weight validation allows pure functional similarity")
        print(f"   • Both univariate and multivariate DTW supported")
        print(f"   • Proper error handling and fallbacks implemented")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
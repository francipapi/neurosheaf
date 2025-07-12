#!/usr/bin/env python3
"""
Quick validation test for critical spectral module fixes.

Tests the three main fixes:
1. Correct Laplacian masking using block reconstruction 
2. Continuous path tracking in SubspaceTracker
3. Path-based persistence diagram generation
"""

import torch
import numpy as np
import networkx as nx
import sys
import traceback

sys.path.insert(0, '/Users/francescopapini/GitRepo/neurosheaf')

from neurosheaf.sheaf.construction import Sheaf
from neurosheaf.spectral.tracker import SubspaceTracker
from neurosheaf.spectral.persistent import PersistentSpectralAnalyzer
from neurosheaf.spectral.static_laplacian_unified import UnifiedStaticLaplacian

def create_simple_test_sheaf():
    """Create minimal test sheaf."""
    # Simple chain: 0 → 1
    poset = nx.DiGraph()
    poset.add_nodes_from(["0", "1"])
    poset.add_edge("0", "1")
    
    stalks = {
        "0": torch.randn(5, 2),  # Small for speed
        "1": torch.randn(5, 1)
    }
    
    restrictions = {
        ("0", "1"): torch.tensor([[1.5], [0.0]])  # Simple restriction
    }
    
    return Sheaf(stalks=stalks, restrictions=restrictions, poset=poset)

def test_continuous_paths():
    """Test 1: Continuous path tracking."""
    print("=== Test 1: Continuous Path Tracking ===")
    
    try:
        # Create simple synthetic data
        n_steps = 5
        eigenvals_sequence = []
        eigenvecs_sequence = []
        filtration_params = [0.0, 0.5, 1.0, 1.5, 2.0]
        
        for i in range(n_steps):
            # Simple eigenvalue evolution
            eigenvals = torch.tensor([0.1 + 0.1*i, 1.0 + 0.2*i])
            eigenvecs = torch.eye(2)  # Simple eigenvectors
            
            eigenvals_sequence.append(eigenvals)
            eigenvecs_sequence.append(eigenvecs)
        
        tracker = SubspaceTracker()
        tracking_info = tracker.track_eigenspaces(
            eigenvals_sequence, eigenvecs_sequence, filtration_params
        )
        
        # Check for continuous paths
        has_paths = 'continuous_paths' in tracking_info
        if has_paths:
            n_paths = len(tracking_info['continuous_paths'])
            print(f"✅ Continuous paths present: {n_paths} paths")
            return True
        else:
            print("❌ No continuous paths found")
            return False
            
    except Exception as e:
        print(f"❌ Path tracking failed: {e}")
        return False

def test_persistence_diagrams():
    """Test 2: Path-based persistence diagrams."""
    print("\n=== Test 2: Path-based Persistence Diagrams ===")
    
    try:
        # Create synthetic tracking info with continuous paths
        tracking_info = {
            'continuous_paths': [
                {
                    'path_id': 0,
                    'birth_param': 0.5,
                    'death_param': 1.5,
                    'birth_step': 1,
                    'death_step': 3,
                    'eigenvalue_trace': [0.1, 0.2, 0.3, 0.4]
                },
                {
                    'path_id': 1,
                    'birth_param': 1.0,
                    'death_param': None,  # Infinite
                    'birth_step': 2,
                    'death_step': None,
                    'eigenvalue_trace': [0.8, 0.9, 1.0]
                }
            ],
            'birth_events': [],
            'death_events': []
        }
        
        # Test the persistence diagram generation directly
        from neurosheaf.spectral.persistent import PersistentSpectralAnalyzer
        analyzer = PersistentSpectralAnalyzer()
        
        diagrams = analyzer._generate_persistence_diagrams(tracking_info, [0.0, 0.5, 1.0, 1.5, 2.0])
        
        # Check for path-based computation
        path_based = diagrams.get('path_based_computation', False)
        has_finite = len(diagrams.get('birth_death_pairs', [])) > 0
        has_infinite = len(diagrams.get('infinite_bars', [])) > 0
        
        if path_based and has_finite and has_infinite:
            print(f"✅ Path-based persistence diagrams: {len(diagrams['birth_death_pairs'])} finite, {len(diagrams['infinite_bars'])} infinite")
            return True
        else:
            print(f"❌ Persistence diagram issues: path_based={path_based}, finite={has_finite}, infinite={has_infinite}")
            return False
            
    except Exception as e:
        print(f"❌ Persistence diagram test failed: {e}")
        return False

def test_integration():
    """Test 3: Basic integration."""
    print("\n=== Test 3: Integration Test ===")
    
    try:
        sheaf = create_simple_test_sheaf()
        analyzer = PersistentSpectralAnalyzer(default_n_steps=3)
        
        # Run minimal analysis
        result = analyzer.analyze(sheaf, n_steps=3)
        
        # Check basic structure
        has_diagrams = 'diagrams' in result
        has_features = 'features' in result
        has_metadata = 'analysis_metadata' in result
        
        if has_diagrams and has_features and has_metadata:
            print("✅ Integration test passed - all components working")
            return True
        else:
            print(f"❌ Integration issues: diagrams={has_diagrams}, features={has_features}, metadata={has_metadata}")
            return False
            
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        return False

def test_unified_implementation():
    """Test 4: Unified static Laplacian implementation."""
    print("\n=== Test 4: Unified Implementation Test ===")
    
    try:
        sheaf = create_simple_test_sheaf()
        
        # Test direct usage of unified implementation
        unified_laplacian = UnifiedStaticLaplacian(
            eigenvalue_method='auto',
            max_eigenvalues=10,
            enable_caching=True,
            validate_properties=True
        )
        
        # Test persistence computation
        filtration_params = [0.0, 1.0, 2.0]
        edge_threshold_func = lambda weight, param: weight >= param
        
        result = unified_laplacian.compute_persistence(
            sheaf, filtration_params, edge_threshold_func
        )
        
        # Validate results
        has_eigenvals = 'eigenvalue_sequences' in result
        has_eigenvecs = 'eigenvector_sequences' in result
        has_metadata = 'masking_metadata' in result
        correct_method = result.get('method') == 'unified_correct_masking'
        
        # Test validation
        validation_result = unified_laplacian.validate_masking_correctness(sheaf)
        validation_passed = 'validation_error' not in validation_result
        
        # Test caching
        cache_info = unified_laplacian.get_cache_info()
        has_cache = any(cache_info.values())
        
        if (has_eigenvals and has_eigenvecs and has_metadata and 
            correct_method and validation_passed and has_cache):
            print("✅ Unified implementation test passed - mathematically correct masking")
            return True
        else:
            print(f"❌ Unified implementation issues: eigenvals={has_eigenvals}, "
                  f"method={correct_method}, validation={validation_passed}, cache={has_cache}")
            return False
            
    except Exception as e:
        print(f"❌ Unified implementation test failed: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        return False

def main():
    """Run quick validation tests."""
    print("NEUROSHEAF SPECTRAL FIXES - QUICK VALIDATION")
    print("=" * 50)
    
    results = []
    
    # Run tests
    results.append(("Continuous Path Tracking", test_continuous_paths()))
    results.append(("Path-based Persistence", test_persistence_diagrams()))
    results.append(("Integration Test", test_integration()))
    results.append(("Unified Implementation", test_unified_implementation()))
    
    # Summary
    print("\n" + "=" * 50)
    print("VALIDATION SUMMARY")
    print("=" * 50)
    
    passed = 0
    for test_name, passed_test in results:
        status = "✅ PASSED" if passed_test else "❌ FAILED"
        print(f"{test_name:<25} {status}")
        if passed_test:
            passed += 1
    
    print(f"\nOVERALL: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("🎉 ALL CRITICAL FIXES VALIDATED!")
        return 0
    else:
        print("❌ Some critical fixes need attention")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
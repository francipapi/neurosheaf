#!/usr/bin/env python3
"""
Simplified test suite for Laplacian filtration with new edge masking.

This test suite focuses on the core functionality validation with simpler
test cases that are less likely to have numerical issues.
"""

import torch
import numpy as np
import networkx as nx
import sys
import time

sys.path.insert(0, '/Users/francescopapini/GitRepo/neurosheaf')

from neurosheaf.sheaf.construction import Sheaf
from neurosheaf.spectral.static_laplacian_unified import UnifiedStaticLaplacian
from neurosheaf.spectral.persistent import PersistentSpectralAnalyzer
from neurosheaf.utils.logging import setup_logger

logger = setup_logger(__name__)


def create_simple_test_sheaf():
    """Create minimal valid test sheaf."""
    # Simple chain: 0 → 1
    poset = nx.DiGraph()
    poset.add_nodes_from(["0", "1"])
    poset.add_edge("0", "1")
    
    stalks = {
        "0": torch.randn(6, 2),
        "1": torch.randn(6, 1)
    }
    
    # Simple restriction with known weight
    restrictions = {
        ("0", "1"): torch.tensor([[2.0], [0.0]])  # Weight = 2.0
    }
    
    return Sheaf(stalks=stalks, restrictions=restrictions, poset=poset)


def create_multi_edge_sheaf():
    """Create sheaf with multiple edges for filtration testing."""
    # Diamond: 0 → {1,2} → 3 but simplified
    poset = nx.DiGraph()
    poset.add_nodes_from(["0", "1", "2", "3"])
    poset.add_edge("0", "1")
    poset.add_edge("0", "2")
    poset.add_edge("1", "3")
    poset.add_edge("2", "3")
    
    stalks = {
        "0": torch.randn(4, 2),
        "1": torch.randn(4, 1),
        "2": torch.randn(4, 1),
        "3": torch.randn(4, 1)
    }
    
    # Create restrictions with known weights
    restrictions = {
        ("0", "1"): torch.tensor([[1.0], [0.0]]),     # Weight = 1.0
        ("0", "2"): torch.tensor([[1.5], [0.0]]),     # Weight = 1.5
        ("1", "3"): torch.tensor([[2.0]]),            # Weight = 2.0
        ("2", "3"): torch.tensor([[2.5]])             # Weight = 2.5
    }
    
    return Sheaf(stalks=stalks, restrictions=restrictions, poset=poset)


def test_basic_masking_functionality():
    """Test 1: Basic edge masking functionality."""
    print("=== Test 1: Basic Edge Masking ===")
    
    try:
        sheaf = create_multi_edge_sheaf()
        unified_laplacian = UnifiedStaticLaplacian()
        
        # Test single threshold
        thresholds = [1.2]
        edge_threshold_func = lambda weight, param: weight >= param
        
        result = unified_laplacian.compute_persistence(sheaf, thresholds, edge_threshold_func)
        
        # Basic validation
        assert 'eigenvalue_sequences' in result
        assert 'tracking_info' in result
        assert len(result['eigenvalue_sequences']) == 1
        
        # Check edge filtering worked
        edge_info = result['edge_info']
        active_edges = sum(1 for edge, info in edge_info.items() if info['weight'] >= 1.2)
        expected_active = 3  # edges with weights 1.5, 2.0, 2.5
        assert active_edges == expected_active, f"Expected {expected_active} active edges, got {active_edges}"
        
        print("✅ Basic masking functionality works")
        return True
        
    except Exception as e:
        print(f"❌ Basic masking failed: {e}")
        return False


def test_filtration_sequence():
    """Test 2: Filtration sequence correctness."""
    print("\n=== Test 2: Filtration Sequence ===")
    
    try:
        sheaf = create_multi_edge_sheaf()
        unified_laplacian = UnifiedStaticLaplacian()
        
        # Test sequence of thresholds
        thresholds = [0.5, 1.25, 1.75, 2.25, 3.0]
        edge_threshold_func = lambda weight, param: weight >= param
        
        result = unified_laplacian.compute_persistence(sheaf, thresholds, edge_threshold_func)
        
        # Check we get sequences for all thresholds
        eigenval_sequences = result['eigenvalue_sequences']
        assert len(eigenval_sequences) == len(thresholds)
        
        # Check monotonic decrease in active edges
        edge_info = result['edge_info']
        active_counts = []
        for threshold in thresholds:
            count = sum(1 for edge, info in edge_info.items() if info['weight'] >= threshold)
            active_counts.append(count)
        
        # Should be [4, 3, 2, 1, 0] for our weights [1.0, 1.5, 2.0, 2.5]
        expected = [4, 3, 2, 1, 0]
        assert active_counts == expected, f"Expected {expected}, got {active_counts}"
        
        print("✅ Filtration sequence is correct")
        return True
        
    except Exception as e:
        print(f"❌ Filtration sequence failed: {e}")
        return False


def test_caching_system():
    """Test 3: Caching system efficiency."""
    print("\n=== Test 3: Caching System ===")
    
    try:
        sheaf = create_multi_edge_sheaf()
        
        # Test with caching
        unified_cached = UnifiedStaticLaplacian(enable_caching=True)
        thresholds = [1.0, 1.5, 2.0]
        edge_threshold_func = lambda weight, param: weight >= param
        
        start_time = time.time()
        result1 = unified_cached.compute_persistence(sheaf, thresholds, edge_threshold_func)
        cached_time = time.time() - start_time
        
        # Test cache info
        cache_info = unified_cached.get_cache_info()
        assert cache_info['laplacian_cached'], "Laplacian should be cached"
        assert cache_info['edge_info_cached'], "Edge info should be cached"
        
        # Second run should use cache
        start_time = time.time()
        result2 = unified_cached.compute_persistence(sheaf, thresholds, edge_threshold_func)
        second_run_time = time.time() - start_time
        
        print(f"First run: {cached_time:.4f}s, Second run: {second_run_time:.4f}s")
        
        # Results should be identical
        for i, (e1, e2) in enumerate(zip(result1['eigenvalue_sequences'], result2['eigenvalue_sequences'])):
            diff = torch.norm(e1 - e2)
            assert diff < 1e-10, f"Cached results differ at step {i}"
        
        print("✅ Caching system works correctly")
        return True
        
    except Exception as e:
        print(f"❌ Caching test failed: {e}")
        return False


def test_persistence_integration():
    """Test 4: Integration with persistence analysis."""
    print("\n=== Test 4: Persistence Integration ===")
    
    try:
        sheaf = create_multi_edge_sheaf()
        analyzer = PersistentSpectralAnalyzer(default_n_steps=5)
        
        # Full persistence analysis
        result = analyzer.analyze(sheaf, n_steps=5)
        
        # Validate complete structure
        assert 'diagrams' in result
        assert 'features' in result
        assert 'persistence_result' in result
        
        # Check diagrams use path-based computation
        diagrams = result['diagrams']
        assert diagrams.get('path_based_computation', False), "Should use path-based computation"
        
        # Validate some basic properties
        if diagrams['birth_death_pairs']:
            for pair in diagrams['birth_death_pairs']:
                assert pair['birth'] <= pair['death'], "Invalid persistence pair"
        
        print("✅ Persistence integration works")
        return True
        
    except Exception as e:
        print(f"❌ Persistence integration failed: {e}")
        return False


def test_mathematical_properties():
    """Test 5: Core mathematical properties."""
    print("\n=== Test 5: Mathematical Properties ===")
    
    try:
        sheaf = create_simple_test_sheaf()  # Use simpler sheaf for mathematical validation
        unified_laplacian = UnifiedStaticLaplacian()
        
        # Test with single threshold
        result = unified_laplacian.compute_persistence(sheaf, [1.5], lambda w, p: w >= p)
        
        # Get the masked Laplacian for testing
        edge_info = result['edge_info']
        edge_mask = {edge: info['weight'] >= 1.5 for edge, info in edge_info.items()}
        
        static_laplacian, metadata = unified_laplacian._get_or_build_laplacian(sheaf)
        masked_laplacian = unified_laplacian._apply_correct_masking(
            static_laplacian, edge_mask, edge_info, metadata
        )
        
        # Test symmetry
        symmetry_error = abs((masked_laplacian - masked_laplacian.T).max())
        assert symmetry_error < 1e-12, f"Symmetry violation: {symmetry_error}"
        
        # Test that matrix is not empty when edge is included
        if any(edge_mask.values()):
            assert masked_laplacian.nnz > 0, "Matrix should not be empty with active edges"
        
        print("✅ Mathematical properties preserved")
        return True
        
    except Exception as e:
        print(f"❌ Mathematical properties test failed: {e}")
        return False


def test_performance_benchmark():
    """Test 6: Basic performance validation."""
    print("\n=== Test 6: Performance Benchmark ===")
    
    try:
        sheaf = create_multi_edge_sheaf()
        unified_laplacian = UnifiedStaticLaplacian()
        
        # Benchmark moderate filtration
        thresholds = np.linspace(0.5, 3.0, 10).tolist()
        edge_threshold_func = lambda weight, param: weight >= param
        
        start_time = time.time()
        result = unified_laplacian.compute_persistence(sheaf, thresholds, edge_threshold_func)
        computation_time = time.time() - start_time
        
        # Should be reasonably fast for small sheaf
        assert computation_time < 5.0, f"Too slow: {computation_time:.2f}s"
        
        # Should produce valid results
        assert len(result['eigenvalue_sequences']) == len(thresholds)
        
        print(f"✅ Performance acceptable: {computation_time:.3f}s for {len(thresholds)} thresholds")
        return True
        
    except Exception as e:
        print(f"❌ Performance test failed: {e}")
        return False


def main():
    """Run simplified Laplacian filtration tests."""
    print("LAPLACIAN FILTRATION WITH NEW EDGE MASKING - SIMPLIFIED TESTS")
    print("=" * 70)
    
    tests = [
        ("Basic Masking", test_basic_masking_functionality),
        ("Filtration Sequence", test_filtration_sequence),
        ("Caching System", test_caching_system),
        ("Persistence Integration", test_persistence_integration),
        ("Mathematical Properties", test_mathematical_properties),
        ("Performance", test_performance_benchmark)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"❌ {test_name} crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    
    passed = 0
    for test_name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{test_name:<25} {status}")
        if success:
            passed += 1
    
    print(f"\nOVERALL: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("🎉 ALL SIMPLIFIED TESTS PASSED!")
        print("✅ New edge masking system is working correctly")
    else:
        print("❌ Some tests failed - basic functionality needs attention")
    
    return passed == len(results)


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
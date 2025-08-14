#!/usr/bin/env python3
"""
Test script for edge cases and performance on large graphs.

This script tests the sheaf inclusion mapper commutative property check 
on various edge cases and larger graphs to ensure robustness and performance.
"""

import torch
import numpy as np
import time
import networkx as nx
from typing import Dict, List, Tuple
import sys
import os

# Add paths for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from neurosheaf.spectral.gw.sheaf_inclusion_mapper import SheafInclusionMapper
from neurosheaf.utils.logging import setup_logger

logger = setup_logger(__name__)


def create_large_restriction_maps(n_edges: int, 
                                 prev_dim: int, 
                                 curr_dim: int,
                                 target_dim: int,
                                 commuting: bool = False) -> Tuple[Dict, Dict]:
    """Create restriction maps for a large graph."""
    prev_restrictions = {}
    curr_restrictions = {}
    
    # Generate edge names
    edges = [(f'layer{i}', f'layer{i+1}') for i in range(n_edges)]
    
    for edge in edges:
        if commuting:
            # Create approximately commuting restrictions
            R_prev = torch.randn(prev_dim, target_dim) * 0.3
            R_curr = torch.zeros(target_dim, curr_dim)
            
            # Make overlapping region similar (approximate commutativity)
            overlap_rows = min(prev_dim, target_dim)
            overlap_cols = min(target_dim, prev_dim)
            
            R_curr[:overlap_rows, :overlap_cols] = R_prev[:overlap_rows, :overlap_cols]
            
            # Add small random values to remaining regions
            if target_dim > overlap_rows:
                R_curr[overlap_rows:, :] = torch.randn(target_dim - overlap_rows, curr_dim) * 0.05
            if curr_dim > overlap_cols:
                R_curr[:, overlap_cols:] = torch.randn(target_dim, curr_dim - overlap_cols) * 0.05
        else:
            # Create random non-commuting restrictions
            R_prev = torch.randn(prev_dim, target_dim)
            R_curr = torch.randn(target_dim, curr_dim)
            
        prev_restrictions[edge] = R_prev
        curr_restrictions[edge] = R_curr
    
    return prev_restrictions, curr_restrictions


def test_large_graph_performance():
    """Test performance with large graphs."""
    logger.info("🧪 Testing large graph performance...")
    
    # Test parameters
    prev_dim, curr_dim = 50, 60
    target_dim = 55
    
    # Test with different graph sizes
    graph_sizes = [10, 50, 100, 500]
    
    results = []
    
    for n_edges in graph_sizes:
        logger.info(f"Testing with {n_edges} edges...")
        
        # Create mapper with sampling enabled for large graphs
        mapper = SheafInclusionMapper(
            method='identity_extension',
            inclusion_quality_tolerance=0.1,
            max_edges_to_check=50  # Sample edges for large graphs
        )
        
        # Create identity inclusion mapping
        inclusion_map = torch.zeros(curr_dim, prev_dim)
        for i in range(prev_dim):
            inclusion_map[i, i] = 1.0
        
        # Create restriction maps
        prev_restrictions, curr_restrictions = create_large_restriction_maps(
            n_edges, prev_dim, curr_dim, target_dim, commuting=False
        )
        
        # Time the commutative property check
        start_time = time.time()
        
        commutes, quality_metadata = mapper._check_commutative_property(
            inclusion_map, prev_restrictions, curr_restrictions
        )
        
        elapsed_time = time.time() - start_time
        
        # Record results
        result = {
            'n_edges': n_edges,
            'elapsed_time': elapsed_time,
            'edges_checked': quality_metadata['edges_checked'],
            'max_residual': quality_metadata['max_residual'],
            'commutes': commutes
        }
        results.append(result)
        
        logger.info(f"  ✅ {n_edges} edges: {elapsed_time:.4f}s, "
                   f"checked {quality_metadata['edges_checked']} edges, "
                   f"max_residual={quality_metadata['max_residual']:.6f}")
    
    # Performance summary
    logger.info("📊 Performance Summary:")
    for result in results:
        logger.info(f"  {result['n_edges']:3d} edges: {result['elapsed_time']:.4f}s "
                   f"({result['edges_checked']:2d} checked)")
    
    return results


def test_edge_cases():
    """Test various edge cases."""
    logger.info("🧪 Testing edge cases...")
    
    mapper = SheafInclusionMapper(method='identity_extension')
    
    test_cases = []
    
    # Case 1: Very small dimensions
    logger.info("Testing very small dimensions...")
    prev_dim, curr_dim = 1, 1
    inclusion_map = torch.ones(curr_dim, prev_dim)
    prev_restrictions = {('a', 'b'): torch.ones(prev_dim, 2)}
    curr_restrictions = {('a', 'b'): torch.ones(2, curr_dim)}
    
    try:
        commutes, metadata = mapper._check_commutative_property(
            inclusion_map, prev_restrictions, curr_restrictions
        )
        test_cases.append(('Small dimensions', True, metadata['max_residual']))
        logger.info(f"  ✅ Small dimensions: commutes={commutes}, residual={metadata['max_residual']:.6f}")
    except Exception as e:
        test_cases.append(('Small dimensions', False, f"Error: {e}"))
        logger.error(f"  ❌ Small dimensions failed: {e}")
    
    # Case 2: Very large dimensions
    logger.info("Testing large dimensions...")
    prev_dim, curr_dim = 200, 250
    inclusion_map = torch.zeros(curr_dim, prev_dim)
    for i in range(prev_dim):
        inclusion_map[i, i] = 1.0
    
    target_dim = 220
    prev_restrictions = {('layer1', 'layer2'): torch.randn(prev_dim, target_dim) * 0.01}
    curr_restrictions = {('layer1', 'layer2'): torch.randn(target_dim, curr_dim) * 0.01}
    
    try:
        start_time = time.time()
        commutes, metadata = mapper._check_commutative_property(
            inclusion_map, prev_restrictions, curr_restrictions
        )
        elapsed = time.time() - start_time
        test_cases.append(('Large dimensions', True, metadata['max_residual']))
        logger.info(f"  ✅ Large dimensions: {elapsed:.4f}s, commutes={commutes}, "
                   f"residual={metadata['max_residual']:.6f}")
    except Exception as e:
        test_cases.append(('Large dimensions', False, f"Error: {e}"))
        logger.error(f"  ❌ Large dimensions failed: {e}")
    
    # Case 3: Extreme aspect ratio (very wide)
    logger.info("Testing extreme aspect ratio (wide)...")
    prev_dim, curr_dim = 5, 10
    target_dim = 100  # Very wide restriction maps
    inclusion_map = torch.zeros(curr_dim, prev_dim)
    for i in range(prev_dim):
        inclusion_map[i, i] = 1.0
    
    prev_restrictions = {('wide', 'map'): torch.randn(prev_dim, target_dim) * 0.1}
    curr_restrictions = {('wide', 'map'): torch.randn(target_dim, curr_dim) * 0.1}
    
    try:
        commutes, metadata = mapper._check_commutative_property(
            inclusion_map, prev_restrictions, curr_restrictions
        )
        test_cases.append(('Wide aspect ratio', True, metadata['max_residual']))
        logger.info(f"  ✅ Wide aspect ratio: commutes={commutes}, "
                   f"residual={metadata['max_residual']:.6f}")
    except Exception as e:
        test_cases.append(('Wide aspect ratio', False, f"Error: {e}"))
        logger.error(f"  ❌ Wide aspect ratio failed: {e}")
    
    # Case 4: Extreme aspect ratio (very tall)  
    logger.info("Testing extreme aspect ratio (tall)...")
    prev_dim, curr_dim = 10, 5
    target_dim = 3  # Very tall restriction maps
    inclusion_map = torch.zeros(curr_dim, prev_dim)
    for i in range(curr_dim):
        inclusion_map[i, i] = 1.0
    
    prev_restrictions = {('tall', 'map'): torch.randn(prev_dim, target_dim) * 0.1}
    curr_restrictions = {('tall', 'map'): torch.randn(target_dim, curr_dim) * 0.1}
    
    try:
        commutes, metadata = mapper._check_commutative_property(
            inclusion_map, prev_restrictions, curr_restrictions
        )
        test_cases.append(('Tall aspect ratio', True, metadata['max_residual']))
        logger.info(f"  ✅ Tall aspect ratio: commutes={commutes}, "
                   f"residual={metadata['max_residual']:.6f}")
    except Exception as e:
        test_cases.append(('Tall aspect ratio', False, f"Error: {e}"))
        logger.error(f"  ❌ Tall aspect ratio failed: {e}")
    
    # Case 5: Zero matrices
    logger.info("Testing zero matrices...")
    prev_dim, curr_dim = 3, 4
    inclusion_map = torch.zeros(curr_dim, prev_dim)
    prev_restrictions = {('zero', 'test'): torch.zeros(prev_dim, 5)}
    curr_restrictions = {('zero', 'test'): torch.zeros(5, curr_dim)}
    
    try:
        commutes, metadata = mapper._check_commutative_property(
            inclusion_map, prev_restrictions, curr_restrictions
        )
        test_cases.append(('Zero matrices', True, metadata['max_residual']))
        logger.info(f"  ✅ Zero matrices: commutes={commutes}, "
                   f"residual={metadata['max_residual']:.6f}")
    except Exception as e:
        test_cases.append(('Zero matrices', False, f"Error: {e}"))
        logger.error(f"  ❌ Zero matrices failed: {e}")
    
    return test_cases


def test_numerical_stability():
    """Test numerical stability with extreme values."""
    logger.info("🧪 Testing numerical stability...")
    
    mapper = SheafInclusionMapper(method='identity_extension', numerical_tolerance=1e-12)
    
    stability_tests = []
    
    # Test with very small values
    logger.info("Testing very small values...")
    prev_dim, curr_dim = 5, 7
    inclusion_map = torch.eye(curr_dim, prev_dim) * 1e-10  # Very small
    prev_restrictions = {('tiny', 'values'): torch.ones(prev_dim, 6) * 1e-15}
    curr_restrictions = {('tiny', 'values'): torch.ones(6, curr_dim) * 1e-15}
    
    try:
        commutes, metadata = mapper._check_commutative_property(
            inclusion_map, prev_restrictions, curr_restrictions
        )
        stability_tests.append(('Very small values', True, metadata['max_residual']))
        logger.info(f"  ✅ Very small values: residual={metadata['max_residual']:.6e}")
    except Exception as e:
        stability_tests.append(('Very small values', False, str(e)))
        logger.error(f"  ❌ Very small values failed: {e}")
    
    # Test with mixed scales
    logger.info("Testing mixed scales...")
    prev_restrictions = {('mixed', 'scale'): torch.cat([
        torch.ones(2, 3) * 1e-10,  # Very small part
        torch.ones(3, 3) * 1e5     # Large part  
    ], dim=0)}
    curr_restrictions = {('mixed', 'scale'): torch.cat([
        torch.ones(3, 3) * 1e-10,  # Very small part
        torch.ones(3, 4) * 1e5     # Large part
    ], dim=1)}
    
    try:
        commutes, metadata = mapper._check_commutative_property(
            inclusion_map, prev_restrictions, curr_restrictions
        )
        stability_tests.append(('Mixed scales', True, metadata['max_residual']))
        logger.info(f"  ✅ Mixed scales: residual={metadata['max_residual']:.6e}")
    except Exception as e:
        stability_tests.append(('Mixed scales', False, str(e)))
        logger.error(f"  ❌ Mixed scales failed: {e}")
    
    return stability_tests


def main():
    """Run all edge case and performance tests."""
    logger.info("🚀 Starting edge cases and large graph testing...")
    
    try:
        # Set random seed for reproducible results
        torch.manual_seed(42)
        np.random.seed(42)
        
        # Test 1: Large graph performance
        perf_results = test_large_graph_performance()
        
        # Test 2: Edge cases
        edge_cases = test_edge_cases()
        
        # Test 3: Numerical stability
        stability_tests = test_numerical_stability()
        
        # Summary
        logger.info("📋 Test Summary:")
        logger.info(f"  Performance tests: {len(perf_results)} completed")
        successful_edge_cases = sum(1 for _, success, _ in edge_cases if success)
        logger.info(f"  Edge case tests: {successful_edge_cases}/{len(edge_cases)} passed")
        successful_stability = sum(1 for _, success, _ in stability_tests if success)
        logger.info(f"  Stability tests: {successful_stability}/{len(stability_tests)} passed")
        
        # Check if performance scales reasonably
        if len(perf_results) >= 2:
            small_time = perf_results[0]['elapsed_time']
            large_time = perf_results[-1]['elapsed_time']
            scaling_factor = large_time / small_time if small_time > 0 else float('inf')
            edge_factor = perf_results[-1]['n_edges'] / perf_results[0]['n_edges']
            
            logger.info(f"  Scaling: {edge_factor:.1f}x edges → {scaling_factor:.1f}x time")
            
            if scaling_factor < edge_factor:
                logger.info("  ✅ Good scaling performance (sublinear)")
            else:
                logger.warning("  ⚠️ Scaling may need optimization")
        
        logger.info("🎉 All tests completed successfully!")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Testing failed: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
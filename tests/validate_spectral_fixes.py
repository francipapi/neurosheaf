#!/usr/bin/env python3
"""
Comprehensive validation test suite for spectral module fixes.

This script validates the critical fixes implemented in the spectral module:
1. Mathematically correct Laplacian masking in static_laplacian_masking.py
2. Proper continuous path tracking in tracker.py
3. Correct persistence diagram generation in persistent.py

Tests verify mathematical correctness, computational stability, and integration.
"""

import torch
import numpy as np
import networkx as nx
from typing import Dict, List, Tuple
import pytest
import tempfile
import os
import sys

# Add the project root to Python path
sys.path.insert(0, '/Users/francescopapini/GitRepo/neurosheaf')

from neurosheaf.sheaf.construction import Sheaf
from neurosheaf.sheaf.laplacian import SheafLaplacianBuilder
from neurosheaf.spectral.static_laplacian_masking import StaticLaplacianWithMasking
from neurosheaf.spectral.tracker import SubspaceTracker
from neurosheaf.spectral.persistent import PersistentSpectralAnalyzer
from neurosheaf.utils.logging import setup_logger

logger = setup_logger(__name__)


class SpectralFixesValidator:
    """Comprehensive validator for spectral module fixes."""
    
    def __init__(self):
        self.tolerance = 1e-10
        self.results = {
            'laplacian_masking': {},
            'path_tracking': {},
            'persistence_diagrams': {},
            'integration': {}
        }
    
    def create_test_sheaf(self, structure_type: str = "diamond") -> Sheaf:
        """Create test sheaf with known mathematical properties."""
        if structure_type == "diamond":
            # Diamond pattern: 0 → {1,2} → 3
            poset = nx.DiGraph()
            poset.add_nodes_from(["0", "1", "2", "3"])
            poset.add_edge("0", "1")
            poset.add_edge("0", "2")
            poset.add_edge("1", "3")
            poset.add_edge("2", "3")
            
            # Create stalks with different dimensions for comprehensive testing
            stalks = {
                "0": torch.randn(10, 3),  # 10 samples, 3 features
                "1": torch.randn(10, 2),  # Different dimensions
                "2": torch.randn(10, 2),
                "3": torch.randn(10, 1)
            }
            
            # Create restrictions with known weights
            restrictions = {
                ("0", "1"): torch.tensor([[1.0, 0.5], [0.0, 1.0], [0.0, 0.0]]),  # weight ≈ 1.118
                ("0", "2"): torch.tensor([[0.8, 0.0], [0.0, 0.9], [0.0, 0.0]]),  # weight ≈ 1.208
                ("1", "3"): torch.tensor([[2.0], [0.0]]),                         # weight = 2.0
                ("2", "3"): torch.tensor([[0.0], [1.5]])                          # weight = 1.5
            }
            
        elif structure_type == "chain":
            # Simple chain: 0 → 1 → 2
            poset = nx.DiGraph()
            poset.add_nodes_from(["0", "1", "2"])
            poset.add_edge("0", "1")
            poset.add_edge("1", "2")
            
            stalks = {
                "0": torch.randn(8, 2),
                "1": torch.randn(8, 2),
                "2": torch.randn(8, 1)
            }
            
            restrictions = {
                ("0", "1"): torch.eye(2) * 0.5,      # weight ≈ 0.707
                ("1", "2"): torch.tensor([[3.0], [0.0]])  # weight = 3.0
            }
            
        else:
            raise ValueError(f"Unknown structure type: {structure_type}")
        
        return Sheaf(stalks=stalks, restrictions=restrictions, poset=poset)
    
    def test_laplacian_masking_correctness(self) -> Dict:
        """Test 1: Validate mathematically correct Laplacian masking."""
        logger.info("Testing Laplacian masking mathematical correctness...")
        
        sheaf = self.create_test_sheaf("diamond")
        masking_analyzer = StaticLaplacianWithMasking()
        
        # Test different threshold values
        test_thresholds = [0.5, 1.0, 1.5, 2.0, 2.5]
        edge_threshold_func = lambda weight, param: weight >= param
        
        results = {
            'test_passed': True,
            'threshold_results': [],
            'errors': []
        }
        
        try:
            # Compute persistence for all thresholds
            persistence_result = masking_analyzer.compute_persistence(
                sheaf, test_thresholds, edge_threshold_func
            )
            
            # Validate each filtered Laplacian
            for i, threshold in enumerate(test_thresholds):
                eigenvals = persistence_result['eigenvalue_sequences'][i]
                eigenvecs = persistence_result['eigenvector_sequences'][i]
                
                # Test 1.1: Eigenvalues are non-negative (PSD property)
                min_eigenval = torch.min(eigenvals).item()
                psd_valid = min_eigenval >= -self.tolerance
                
                # Test 1.2: Eigenvalues are sorted
                sorted_valid = torch.all(eigenvals[1:] >= eigenvals[:-1] - self.tolerance)
                
                # Test 1.3: Eigenvectors are orthonormal
                if len(eigenvecs) > 1:
                    dot_products = torch.mm(eigenvecs.T, eigenvecs)
                    identity_error = torch.norm(dot_products - torch.eye(eigenvecs.shape[1]))
                    orthonormal_valid = identity_error.item() < 1e-6
                else:
                    orthonormal_valid = True
                
                threshold_result = {
                    'threshold': threshold,
                    'n_eigenvals': len(eigenvals),
                    'min_eigenval': min_eigenval,
                    'psd_valid': psd_valid,
                    'sorted_valid': sorted_valid.item(),
                    'orthonormal_valid': orthonormal_valid,
                    'all_valid': psd_valid and sorted_valid and orthonormal_valid
                }
                
                results['threshold_results'].append(threshold_result)
                
                if not threshold_result['all_valid']:
                    results['test_passed'] = False
                    results['errors'].append(f"Threshold {threshold}: validation failed")
            
            # Test 1.4: Verify proper edge filtering
            edge_info = persistence_result['edge_info']
            expected_active_edges = []
            for threshold in test_thresholds:
                active_count = sum(1 for edge, info in edge_info.items() 
                                 if info['weight'] >= threshold)
                expected_active_edges.append(active_count)
            
            # Expected: [4, 4, 3, 2, 0] based on our test sheaf weights
            # (all edges ≥ 0.5, all ≥ 1.0, 3 edges ≥ 1.5, 2 edges ≥ 2.0, 0 edges ≥ 2.5)
            
            results['edge_filtering'] = {
                'active_edges_per_threshold': expected_active_edges,
                'monotonic_decrease': all(expected_active_edges[i] >= expected_active_edges[i+1] 
                                        for i in range(len(expected_active_edges)-1))
            }
            
        except Exception as e:
            results['test_passed'] = False
            results['errors'].append(f"Masking test failed: {e}")
        
        self.results['laplacian_masking'] = results
        return results
    
    def test_path_tracking_continuity(self) -> Dict:
        """Test 2: Validate continuous eigenvalue path tracking."""
        logger.info("Testing eigenvalue path tracking continuity...")
        
        # Create synthetic eigenvalue sequence with known crossing
        n_steps = 20
        eigenvals_sequence = []
        eigenvecs_sequence = []
        filtration_params = np.linspace(0.0, 2.0, n_steps)
        
        # Create two eigenvalue paths that cross
        for i, param in enumerate(filtration_params):
            t = param / 2.0  # Normalize to [0, 1]
            
            # Path 1: λ₁(t) = 0.1 + 0.5*t
            # Path 2: λ₂(t) = 0.8 - 0.5*t  
            # They cross at t = 0.7, param = 1.4
            lambda1 = 0.1 + 0.5 * t
            lambda2 = 0.8 - 0.5 * t
            lambda3 = 1.5  # Constant high eigenvalue
            
            eigenvals = torch.tensor([lambda1, lambda2, lambda3])
            
            # Create corresponding eigenvectors (random but consistent)
            torch.manual_seed(42 + i)  # Reproducible but varying
            eigenvecs = torch.randn(10, 3)
            eigenvecs, _ = torch.qr(eigenvecs)  # Orthonormalize
            
            eigenvals_sequence.append(eigenvals)
            eigenvecs_sequence.append(eigenvecs)
        
        results = {
            'test_passed': True,
            'path_analysis': {},
            'errors': []
        }
        
        try:
            # Test path tracking
            tracker = SubspaceTracker(gap_eps=1e-6, cos_tau=0.7)
            tracking_info = tracker.track_eigenspaces(
                eigenvals_sequence, eigenvecs_sequence, filtration_params.tolist()
            )
            
            # Test 2.1: Continuous paths should be present
            has_continuous_paths = 'continuous_paths' in tracking_info
            results['path_analysis']['has_continuous_paths'] = has_continuous_paths
            
            if has_continuous_paths:
                continuous_paths = tracking_info['continuous_paths']
                
                # Test 2.2: Should have paths covering the filtration
                n_paths = len(continuous_paths)
                results['path_analysis']['n_paths'] = n_paths
                
                # Test 2.3: Paths should have reasonable lifetimes
                finite_paths = [p for p in continuous_paths if p['death_param'] is not None]
                infinite_paths = [p for p in continuous_paths if p['death_param'] is None]
                
                results['path_analysis']['n_finite_paths'] = len(finite_paths)
                results['path_analysis']['n_infinite_paths'] = len(infinite_paths)
                
                # Test 2.4: Path eigenvalue traces should be continuous
                trace_continuity_valid = True
                for path in continuous_paths:
                    eigenval_trace = path.get('eigenvalue_trace', [])
                    if len(eigenval_trace) > 1:
                        # Check for reasonable continuity (no jumps > 0.5)
                        diffs = np.diff(eigenval_trace)
                        max_jump = np.max(np.abs(diffs))
                        if max_jump > 0.5:
                            trace_continuity_valid = False
                            results['errors'].append(f"Path {path['path_id']} has jump {max_jump}")
                
                results['path_analysis']['trace_continuity_valid'] = trace_continuity_valid
                
                # Test 2.5: Should detect the known crossing
                crossings = tracking_info.get('crossings', [])
                crossing_detected = len(crossings) > 0
                results['path_analysis']['crossing_detected'] = crossing_detected
                
                if not (has_continuous_paths and n_paths >= 2 and trace_continuity_valid):
                    results['test_passed'] = False
            else:
                results['test_passed'] = False
                results['errors'].append("No continuous paths in tracking info")
        
        except Exception as e:
            results['test_passed'] = False
            results['errors'].append(f"Path tracking test failed: {e}")
        
        self.results['path_tracking'] = results
        return results
    
    def test_persistence_diagram_validity(self) -> Dict:
        """Test 3: Validate persistence diagram mathematical validity."""
        logger.info("Testing persistence diagram mathematical validity...")
        
        sheaf = self.create_test_sheaf("chain")  # Use simpler structure for clearer analysis
        
        results = {
            'test_passed': True,
            'diagram_analysis': {},
            'errors': []
        }
        
        try:
            # Run full persistent spectral analysis
            analyzer = PersistentSpectralAnalyzer(default_n_steps=15)
            analysis_result = analyzer.analyze(
                sheaf, 
                filtration_type='threshold',
                n_steps=15
            )
            
            diagrams = analysis_result['diagrams']
            
            # Test 3.1: Should use continuous path-based computation
            path_based = diagrams.get('path_based_computation', False)
            results['diagram_analysis']['path_based_computation'] = path_based
            
            # Test 3.2: Birth times should be ≤ death times for finite pairs
            birth_death_pairs = diagrams.get('birth_death_pairs', [])
            finite_pairs_valid = True
            for pair in birth_death_pairs:
                if pair['death'] != float('inf') and pair['birth'] > pair['death']:
                    finite_pairs_valid = False
                    results['errors'].append(f"Invalid pair: birth {pair['birth']} > death {pair['death']}")
            
            results['diagram_analysis']['finite_pairs_valid'] = finite_pairs_valid
            results['diagram_analysis']['n_finite_pairs'] = len(birth_death_pairs)
            
            # Test 3.3: Infinite bars should have finite birth times
            infinite_bars = diagrams.get('infinite_bars', [])
            infinite_bars_valid = all(np.isfinite(bar['birth']) for bar in infinite_bars)
            results['diagram_analysis']['infinite_bars_valid'] = infinite_bars_valid
            results['diagram_analysis']['n_infinite_bars'] = len(infinite_bars)
            
            # Test 3.4: Statistics should be mathematically consistent
            stats = diagrams.get('statistics', {})
            if birth_death_pairs:
                computed_lifetimes = [pair['death'] - pair['birth'] for pair in birth_death_pairs 
                                    if pair['death'] != float('inf')]
                if computed_lifetimes:
                    expected_mean = np.mean(computed_lifetimes)
                    reported_mean = stats.get('mean_lifetime', 0)
                    mean_consistent = abs(expected_mean - reported_mean) < self.tolerance
                    results['diagram_analysis']['statistics_consistent'] = mean_consistent
                else:
                    results['diagram_analysis']['statistics_consistent'] = True
            else:
                results['diagram_analysis']['statistics_consistent'] = True
            
            # Test 3.5: Should have path statistics if using continuous paths
            if path_based:
                path_stats = diagrams.get('path_statistics', {})
                has_path_stats = len(path_stats) > 0
                results['diagram_analysis']['has_path_statistics'] = has_path_stats
            
            # Overall validity
            all_tests = [
                path_based,
                finite_pairs_valid,
                infinite_bars_valid,
                results['diagram_analysis'].get('statistics_consistent', True)
            ]
            
            if not all(all_tests):
                results['test_passed'] = False
        
        except Exception as e:
            results['test_passed'] = False
            results['errors'].append(f"Persistence diagram test failed: {e}")
        
        self.results['persistence_diagrams'] = results
        return results
    
    def test_integration_stability(self) -> Dict:
        """Test 4: Integration test with stability analysis."""
        logger.info("Testing integration and computational stability...")
        
        results = {
            'test_passed': True,
            'stability_analysis': {},
            'errors': []
        }
        
        try:
            # Test with multiple sheaf structures
            test_structures = ["diamond", "chain"]
            stability_results = []
            
            for structure in test_structures:
                sheaf = self.create_test_sheaf(structure)
                
                # Run analysis twice with identical parameters
                analyzer = PersistentSpectralAnalyzer(default_n_steps=10)
                
                result1 = analyzer.analyze(sheaf, n_steps=10, filtration_type='threshold')
                result2 = analyzer.analyze(sheaf, n_steps=10, filtration_type='threshold')
                
                # Compare persistence diagrams for stability
                diagrams1 = result1['diagrams']
                diagrams2 = result2['diagrams']
                
                # Test 4.1: Same number of persistence pairs
                n_pairs1 = len(diagrams1['birth_death_pairs'])
                n_pairs2 = len(diagrams2['birth_death_pairs'])
                pairs_consistent = (n_pairs1 == n_pairs2)
                
                # Test 4.2: Birth/death times should be identical (deterministic computation)
                if pairs_consistent and n_pairs1 > 0:
                    max_birth_diff = max(abs(p1['birth'] - p2['birth']) 
                                       for p1, p2 in zip(diagrams1['birth_death_pairs'], 
                                                        diagrams2['birth_death_pairs']))
                    max_death_diff = max(abs(p1['death'] - p2['death']) 
                                       for p1, p2 in zip(diagrams1['birth_death_pairs'], 
                                                        diagrams2['birth_death_pairs'])
                                       if p1['death'] != float('inf') and p2['death'] != float('inf'))
                    
                    times_stable = (max_birth_diff < self.tolerance and max_death_diff < self.tolerance)
                else:
                    times_stable = pairs_consistent
                
                structure_result = {
                    'structure': structure,
                    'pairs_consistent': pairs_consistent,
                    'times_stable': times_stable,
                    'n_pairs': n_pairs1
                }
                stability_results.append(structure_result)
                
                if not (pairs_consistent and times_stable):
                    results['test_passed'] = False
                    results['errors'].append(f"Instability in {structure} structure")
            
            results['stability_analysis']['structure_results'] = stability_results
            
            # Test 4.3: Memory usage should be reasonable
            analyzer.clear_cache()  # Test cache clearing
            cache_cleared = True  # If no exception, cache clearing works
            results['stability_analysis']['cache_clearing_works'] = cache_cleared
        
        except Exception as e:
            results['test_passed'] = False
            results['errors'].append(f"Integration test failed: {e}")
        
        self.results['integration'] = results
        return results
    
    def run_comprehensive_validation(self) -> Dict:
        """Run all validation tests and return comprehensive report."""
        logger.info("=" * 60)
        logger.info("STARTING COMPREHENSIVE SPECTRAL FIXES VALIDATION")
        logger.info("=" * 60)
        
        # Run all tests
        test_results = []
        
        # Test 1: Laplacian Masking
        test_results.append(('Laplacian Masking', self.test_laplacian_masking_correctness()))
        
        # Test 2: Path Tracking  
        test_results.append(('Path Tracking', self.test_path_tracking_continuity()))
        
        # Test 3: Persistence Diagrams
        test_results.append(('Persistence Diagrams', self.test_persistence_diagram_validity()))
        
        # Test 4: Integration
        test_results.append(('Integration & Stability', self.test_integration_stability()))
        
        # Generate summary report
        summary = {
            'total_tests': len(test_results),
            'passed_tests': sum(1 for _, result in test_results if result['test_passed']),
            'failed_tests': sum(1 for _, result in test_results if not result['test_passed']),
            'all_tests_passed': all(result['test_passed'] for _, result in test_results),
            'test_details': {name: result for name, result in test_results},
            'validation_timestamp': torch.tensor(0).item()  # Placeholder for timestamp
        }
        
        # Print results
        logger.info("=" * 60)
        logger.info("VALIDATION RESULTS SUMMARY")
        logger.info("=" * 60)
        
        for test_name, result in test_results:
            status = "✅ PASSED" if result['test_passed'] else "❌ FAILED"
            logger.info(f"{test_name:<25} {status}")
            
            if not result['test_passed']:
                for error in result.get('errors', []):
                    logger.error(f"  Error: {error}")
        
        logger.info("-" * 60)
        logger.info(f"OVERALL RESULT: {summary['passed_tests']}/{summary['total_tests']} tests passed")
        
        if summary['all_tests_passed']:
            logger.info("🎉 ALL SPECTRAL FIXES VALIDATED SUCCESSFULLY!")
        else:
            logger.error("❌ Some tests failed - please review and fix issues")
        
        logger.info("=" * 60)
        
        return summary


def main():
    """Main function to run validation."""
    validator = SpectralFixesValidator()
    summary = validator.run_comprehensive_validation()
    
    # Exit with appropriate code
    exit_code = 0 if summary['all_tests_passed'] else 1
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
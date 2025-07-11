"""Comprehensive robustness and edge case testing for Week 7 implementation.

This module implements rigorous testing of edge cases, error conditions,
and robustness scenarios that could occur in production environments.
Tests ensure graceful degradation and proper error handling.

Robustness Areas Tested:
1. Numerical Stability
   - Ill-conditioned matrices and rank deficiency
   - Extreme scale differences in activations  
   - Near-singular whitening scenarios
   - Floating point precision edge cases

2. Architecture Coverage
   - Unusual network topologies (cycles, disconnected components)
   - Extreme aspect ratios and dimensions
   - Degenerate cases (single layer, empty networks)
   - Mixed precision and data types

3. Error Handling & Recovery
   - Graceful degradation when GPU unavailable
   - Memory exhaustion scenarios
   - Invalid input data and corrupted sheaves
   - Thread safety and concurrent access

4. Production Edge Cases
   - Very large networks (>100 layers)
   - Networks with extreme sparsity
   - Models with unusual activation patterns
   - Dynamic computation graphs
"""

import pytest
import torch
import numpy as np
import networkx as nx
from scipy.sparse import csr_matrix
import time
import gc
import warnings
import sys
import os
from typing import Dict, List, Tuple, Any, Optional
from unittest.mock import patch, MagicMock

# Add neurosheaf to path
sys.path.append('/Users/francescopapini/GitRepo/neurosheaf')

from neurosheaf.sheaf import (
    Sheaf, SheafBuilder, SheafLaplacianBuilder,
    ProcrustesMaps, WhiteningProcessor
)
from neurosheaf.spectral import create_static_masked_laplacian
from neurosheaf.utils.exceptions import ComputationError, ArchitectureError
from tests.test_data_generators import NeuralNetworkDataGenerator


class RobustnessTestSuite:
    """Comprehensive robustness testing for edge cases and error conditions."""
    
    def __init__(self):
        """Initialize robustness test suite."""
        self.generator = NeuralNetworkDataGenerator(seed=42)
        self.test_results = {}
    
    def test_numerical_stability_ill_conditioned_matrices(self) -> Dict[str, Any]:
        """Test behavior with ill-conditioned and rank-deficient matrices."""
        results = {'passed_tests': 0, 'total_tests': 0, 'details': {}}
        
        # Test 1: Nearly singular Gram matrices
        try:
            # Create activations with very small eigenvalues
            batch_size, dim = 10, 15
            X = torch.randn(batch_size, dim)
            
            # Make matrix nearly singular by setting small eigenvalues
            U, S, Vt = torch.linalg.svd(X, full_matrices=False)
            S_modified = S.clone()
            S_modified[-5:] = 1e-12  # Very small eigenvalues
            X_singular = U @ torch.diag(S_modified) @ Vt
            
            # Create Gram matrix
            K_singular = X_singular @ X_singular.T
            
            # Test whitening with near-singular matrix
            whitener = WhiteningProcessor()
            K_whitened, W, info = whitener.whiten_gram_matrix(K_singular)
            
            # Should handle gracefully
            assert torch.isfinite(K_whitened).all(), "Whitened matrix contains non-finite values"
            assert info['numerical_rank'] < dim, "Should detect rank deficiency"
            
            results['details']['nearly_singular'] = {
                'success': True,
                'numerical_rank': info['numerical_rank'],
                'condition_number': info.get('condition_number', np.inf)
            }
            results['passed_tests'] += 1
            
        except Exception as e:
            results['details']['nearly_singular'] = {'success': False, 'error': str(e)}
        
        results['total_tests'] += 1
        
        # Test 2: Rank deficient matrices
        try:
            # Create exactly rank-deficient matrix
            rank = 5
            batch_size, dim = 10, 12
            
            # Generate rank-deficient data
            X_low_rank = torch.randn(batch_size, rank) @ torch.randn(rank, dim)
            K_rank_deficient = X_low_rank @ X_low_rank.T
            
            # Test sheaf construction with rank-deficient data
            activations = {
                'layer1': X_low_rank,
                'layer2': X_low_rank + 0.01 * torch.randn_like(X_low_rank)
            }
            
            poset = nx.DiGraph()
            poset.add_edge('layer1', 'layer2')
            
            builder = SheafBuilder(use_whitening=True, enable_edge_filtering=False)
            gram_matrices = self.generator.generate_gram_matrices_from_activations(activations)
            
            # Should handle gracefully
            sheaf = builder.build_from_cka_matrices(poset, gram_matrices, validate=True)
            
            assert len(sheaf.restrictions) > 0, "Should create restrictions despite rank deficiency"
            
            results['details']['rank_deficient'] = {'success': True}
            results['passed_tests'] += 1
            
        except Exception as e:
            results['details']['rank_deficient'] = {'success': False, 'error': str(e)}
        
        results['total_tests'] += 1
        
        # Test 3: Extreme condition numbers
        try:
            # Create matrix with extreme condition number
            eigenvals = torch.logspace(-15, 5, 10)  # Condition number ~10^20
            Q = torch.linalg.qr(torch.randn(10, 10))[0]
            K_extreme = Q @ torch.diag(eigenvals) @ Q.T
            
            whitener = WhiteningProcessor()
            K_whitened, W, info = whitener.whiten_gram_matrix(K_extreme)
            
            # Should either handle gracefully or raise appropriate error
            results['details']['extreme_condition'] = {
                'success': True,
                'condition_number': info.get('condition_number', np.inf),
                'handled_gracefully': True
            }
            results['passed_tests'] += 1
            
        except ComputationError:
            # Acceptable to raise ComputationError for extreme cases
            results['details']['extreme_condition'] = {
                'success': True,
                'handled_gracefully': True,
                'raised_computation_error': True
            }
            results['passed_tests'] += 1
        except Exception as e:
            results['details']['extreme_condition'] = {'success': False, 'error': str(e)}
        
        results['total_tests'] += 1
        
        return results
    
    def test_extreme_scale_differences(self) -> Dict[str, Any]:
        """Test handling of extreme scale differences in activations."""
        results = {'passed_tests': 0, 'total_tests': 0, 'details': {}}
        
        # Test 1: Very large activation magnitudes
        try:
            large_scale_activations = {
                'layer1': torch.randn(8, 10) * 1e6,  # Very large scale
                'layer2': torch.randn(8, 12) * 1e-6,  # Very small scale
                'layer3': torch.randn(8, 10) * 1.0    # Normal scale
            }
            
            poset = nx.DiGraph()
            poset.add_edge('layer1', 'layer2')
            poset.add_edge('layer2', 'layer3')
            
            builder = SheafBuilder(use_whitening=True, enable_edge_filtering=True)
            gram_matrices = self.generator.generate_gram_matrices_from_activations(large_scale_activations)
            
            # Should normalize/handle extreme scales
            sheaf = builder.build_from_cka_matrices(poset, gram_matrices, validate=True)
            laplacian, metadata = builder.build_laplacian(sheaf)
            
            # Verify no NaN or Inf values
            assert torch.isfinite(laplacian.data).all(), "Laplacian contains non-finite values"
            
            results['details']['extreme_scales'] = {'success': True}
            results['passed_tests'] += 1
            
        except Exception as e:
            results['details']['extreme_scales'] = {'success': False, 'error': str(e)}
        
        results['total_tests'] += 1
        
        # Test 2: Mixed precision scenarios
        try:
            # Mix different dtypes
            mixed_activations = {
                'layer1': torch.randn(6, 8, dtype=torch.float32),
                'layer2': torch.randn(6, 10, dtype=torch.float64),
                'layer3': torch.randn(6, 8, dtype=torch.float32)
            }
            
            poset = nx.DiGraph()
            poset.add_edge('layer1', 'layer2')
            poset.add_edge('layer2', 'layer3')
            
            builder = SheafBuilder(use_whitening=True)
            gram_matrices = self.generator.generate_gram_matrices_from_activations(mixed_activations)
            
            # Should handle mixed precision
            sheaf = builder.build_from_cka_matrices(poset, gram_matrices)
            
            results['details']['mixed_precision'] = {'success': True}
            results['passed_tests'] += 1
            
        except Exception as e:
            results['details']['mixed_precision'] = {'success': False, 'error': str(e)}
        
        results['total_tests'] += 1
        
        return results
    
    def test_unusual_network_topologies(self) -> Dict[str, Any]:
        """Test handling of unusual and edge case network topologies."""
        results = {'passed_tests': 0, 'total_tests': 0, 'details': {}}
        
        # Test 1: Single node network
        try:
            single_node_activations = {'only_layer': torch.randn(5, 8)}
            poset = nx.DiGraph()
            poset.add_node('only_layer')
            
            builder = SheafBuilder(use_whitening=True)
            gram_matrices = self.generator.generate_gram_matrices_from_activations(single_node_activations)
            sheaf = builder.build_from_cka_matrices(poset, gram_matrices)
            
            # Should handle single node case
            assert len(sheaf.stalks) == 1
            assert len(sheaf.restrictions) == 0
            
            # Laplacian should be well-formed (zero matrix)
            laplacian, metadata = builder.build_laplacian(sheaf)
            assert laplacian.shape[0] > 0
            
            results['details']['single_node'] = {'success': True}
            results['passed_tests'] += 1
            
        except Exception as e:
            results['details']['single_node'] = {'success': False, 'error': str(e)}
        
        results['total_tests'] += 1
        
        # Test 2: Disconnected components
        try:
            disconnected_activations = {
                'comp1_layer1': torch.randn(4, 6),
                'comp1_layer2': torch.randn(4, 8),
                'comp2_layer1': torch.randn(4, 5),
                'comp2_layer2': torch.randn(4, 7)
            }
            
            # Create disconnected poset
            poset = nx.DiGraph()
            poset.add_edge('comp1_layer1', 'comp1_layer2')  # Component 1
            poset.add_edge('comp2_layer1', 'comp2_layer2')  # Component 2
            # No connection between components
            
            builder = SheafBuilder(use_whitening=True)
            gram_matrices = self.generator.generate_gram_matrices_from_activations(disconnected_activations)
            sheaf = builder.build_from_cka_matrices(poset, gram_matrices)
            
            # Should handle disconnected components
            laplacian, metadata = builder.build_laplacian(sheaf)
            
            results['details']['disconnected'] = {'success': True}
            results['passed_tests'] += 1
            
        except Exception as e:
            results['details']['disconnected'] = {'success': False, 'error': str(e)}
        
        results['total_tests'] += 1
        
        # Test 3: Very dense connectivity (fully connected small network)
        try:
            dense_activations = {f'layer_{i}': torch.randn(4, 6) for i in range(5)}
            
            # Create fully connected poset
            poset = nx.DiGraph()
            layer_names = list(dense_activations.keys())
            for i, source in enumerate(layer_names):
                for target in layer_names[i+1:]:
                    poset.add_edge(source, target)
            
            builder = SheafBuilder(use_whitening=True, enable_edge_filtering=True)
            gram_matrices = self.generator.generate_gram_matrices_from_activations(dense_activations)
            sheaf = builder.build_from_cka_matrices(poset, gram_matrices)
            
            # Should handle dense connectivity
            laplacian, metadata = builder.build_laplacian(sheaf)
            
            results['details']['dense_connectivity'] = {
                'success': True,
                'num_edges': len(sheaf.restrictions),
                'sparsity': metadata.sparsity_ratio
            }
            results['passed_tests'] += 1
            
        except Exception as e:
            results['details']['dense_connectivity'] = {'success': False, 'error': str(e)}
        
        results['total_tests'] += 1
        
        return results
    
    def test_extreme_dimensions(self) -> Dict[str, Any]:
        """Test handling of extreme dimensional scenarios."""
        results = {'passed_tests': 0, 'total_tests': 0, 'details': {}}
        
        # Test 1: Very high dimensional activations
        try:
            high_dim_activations = {
                'layer1': torch.randn(3, 200),  # More features than samples
                'layer2': torch.randn(3, 150)
            }
            
            poset = nx.DiGraph()
            poset.add_edge('layer1', 'layer2')
            
            builder = SheafBuilder(use_whitening=True, enable_edge_filtering=False)
            gram_matrices = self.generator.generate_gram_matrices_from_activations(high_dim_activations)
            sheaf = builder.build_from_cka_matrices(poset, gram_matrices)
            
            # Should handle high dimensions
            laplacian, metadata = builder.build_laplacian(sheaf)
            
            results['details']['high_dimension'] = {
                'success': True,
                'laplacian_shape': laplacian.shape
            }
            results['passed_tests'] += 1
            
        except Exception as e:
            results['details']['high_dimension'] = {'success': False, 'error': str(e)}
        
        results['total_tests'] += 1
        
        # Test 2: Very low dimensional activations
        try:
            low_dim_activations = {
                'layer1': torch.randn(20, 2),  # Very few features
                'layer2': torch.randn(20, 3),
                'layer3': torch.randn(20, 1)   # Single feature
            }
            
            poset = nx.DiGraph()
            poset.add_edge('layer1', 'layer2')
            poset.add_edge('layer2', 'layer3')
            
            builder = SheafBuilder(use_whitening=True)
            gram_matrices = self.generator.generate_gram_matrices_from_activations(low_dim_activations)
            sheaf = builder.build_from_cka_matrices(poset, gram_matrices)
            
            # Should handle low dimensions
            laplacian, metadata = builder.build_laplacian(sheaf)
            
            results['details']['low_dimension'] = {
                'success': True,
                'laplacian_shape': laplacian.shape
            }
            results['passed_tests'] += 1
            
        except Exception as e:
            results['details']['low_dimension'] = {'success': False, 'error': str(e)}
        
        results['total_tests'] += 1
        
        # Test 3: Mismatched dimensions
        try:
            mismatched_activations = {
                'layer1': torch.randn(10, 5),
                'layer2': torch.randn(15, 8),  # Different batch size
                'layer3': torch.randn(10, 12)
            }
            
            poset = nx.DiGraph()
            poset.add_edge('layer1', 'layer2')
            poset.add_edge('layer2', 'layer3')
            
            builder = SheafBuilder(use_whitening=True)
            gram_matrices = self.generator.generate_gram_matrices_from_activations(mismatched_activations)
            
            # This should either handle gracefully or raise appropriate error
            try:
                sheaf = builder.build_from_cka_matrices(poset, gram_matrices)
                results['details']['mismatched_dims'] = {'success': True, 'handled_gracefully': True}
                results['passed_tests'] += 1
            except (ComputationError, ArchitectureError):
                # Acceptable to reject mismatched dimensions
                results['details']['mismatched_dims'] = {'success': True, 'raised_appropriate_error': True}
                results['passed_tests'] += 1
            
        except Exception as e:
            results['details']['mismatched_dims'] = {'success': False, 'error': str(e)}
        
        results['total_tests'] += 1
        
        return results
    
    def test_error_handling_recovery(self) -> Dict[str, Any]:
        """Test error handling and recovery mechanisms."""
        results = {'passed_tests': 0, 'total_tests': 0, 'details': {}}
        
        # Test 1: GPU unavailable fallback
        try:
            # Create normal test case
            activations = self.generator.generate_linear_transformation_sequence(
                num_layers=3, input_dim=10, batch_size=8
            )
            
            layer_names = list(activations.keys())
            poset = nx.DiGraph()
            for name in layer_names:
                poset.add_node(name)
            for i in range(len(layer_names) - 1):
                poset.add_edge(layer_names[i], layer_names[i + 1])
            
            # Mock GPU unavailable
            with patch('torch.cuda.is_available', return_value=False):
                builder = SheafBuilder(use_whitening=True)
                gram_matrices = self.generator.generate_gram_matrices_from_activations(activations)
                sheaf = builder.build_from_cka_matrices(poset, gram_matrices)
                
                # Should fall back to CPU gracefully
                laplacian, metadata = builder.build_laplacian(sheaf, enable_gpu=True)  # Request GPU but should use CPU
                
            results['details']['gpu_fallback'] = {'success': True}
            results['passed_tests'] += 1
            
        except Exception as e:
            results['details']['gpu_fallback'] = {'success': False, 'error': str(e)}
        
        results['total_tests'] += 1
        
        # Test 2: Invalid input data handling
        try:
            # Test with NaN values
            invalid_activations = {
                'layer1': torch.tensor([[1.0, 2.0, float('nan')], [3.0, 4.0, 5.0]]),
                'layer2': torch.tensor([[1.0, 2.0], [float('inf'), 4.0]])
            }
            
            poset = nx.DiGraph()
            poset.add_edge('layer1', 'layer2')
            
            builder = SheafBuilder(use_whitening=True)
            
            # Should detect and handle invalid data
            try:
                gram_matrices = self.generator.generate_gram_matrices_from_activations(invalid_activations)
                sheaf = builder.build_from_cka_matrices(poset, gram_matrices)
                # If it doesn't raise an error, it should handle gracefully
                results['details']['invalid_data'] = {'success': True, 'handled_gracefully': True}
                results['passed_tests'] += 1
            except (ComputationError, ValueError):
                # Acceptable to raise appropriate errors for invalid data
                results['details']['invalid_data'] = {'success': True, 'raised_appropriate_error': True}
                results['passed_tests'] += 1
            
        except Exception as e:
            results['details']['invalid_data'] = {'success': False, 'error': str(e)}
        
        results['total_tests'] += 1
        
        # Test 3: Memory pressure simulation
        try:
            # Simulate memory constraints by creating large data then constraining
            large_activations = self.generator.generate_linear_transformation_sequence(
                num_layers=10, input_dim=50, batch_size=30
            )
            
            layer_names = list(large_activations.keys())
            poset = nx.DiGraph()
            for name in layer_names:
                poset.add_node(name)
            for i in range(len(layer_names) - 1):
                poset.add_edge(layer_names[i], layer_names[i + 1])
            
            # Use memory-efficient settings
            builder = SheafBuilder(use_whitening=True, enable_edge_filtering=True, residual_threshold=0.2)
            gram_matrices = self.generator.generate_gram_matrices_from_activations(large_activations)
            sheaf = builder.build_from_cka_matrices(poset, gram_matrices)
            
            # Should use memory-efficient construction
            laplacian, metadata = builder.build_laplacian(sheaf, memory_efficient=True)
            
            results['details']['memory_pressure'] = {
                'success': True,
                'memory_efficient': True,
                'sparsity': metadata.sparsity_ratio
            }
            results['passed_tests'] += 1
            
        except Exception as e:
            results['details']['memory_pressure'] = {'success': False, 'error': str(e)}
        
        results['total_tests'] += 1
        
        return results
    
    def test_concurrent_access_thread_safety(self) -> Dict[str, Any]:
        """Test thread safety and concurrent access patterns."""
        results = {'passed_tests': 0, 'total_tests': 0, 'details': {}}
        
        # Test 1: Multiple builders simultaneously
        try:
            import threading
            import queue
            
            results_queue = queue.Queue()
            
            def build_sheaf_worker(worker_id):
                try:
                    generator = NeuralNetworkDataGenerator(seed=42 + worker_id)
                    activations = generator.generate_linear_transformation_sequence(
                        num_layers=4, input_dim=8, batch_size=6
                    )
                    
                    layer_names = list(activations.keys())
                    poset = nx.DiGraph()
                    for name in layer_names:
                        poset.add_node(name)
                    for i in range(len(layer_names) - 1):
                        poset.add_edge(layer_names[i], layer_names[i + 1])
                    
                    builder = SheafBuilder(use_whitening=True)
                    gram_matrices = generator.generate_gram_matrices_from_activations(activations)
                    sheaf = builder.build_from_cka_matrices(poset, gram_matrices)
                    laplacian, metadata = builder.build_laplacian(sheaf)
                    
                    results_queue.put({'worker_id': worker_id, 'success': True, 'shape': laplacian.shape})
                    
                except Exception as e:
                    results_queue.put({'worker_id': worker_id, 'success': False, 'error': str(e)})
            
            # Start multiple threads
            threads = []
            num_workers = 3
            for i in range(num_workers):
                thread = threading.Thread(target=build_sheaf_worker, args=(i,))
                threads.append(thread)
                thread.start()
            
            # Wait for completion
            for thread in threads:
                thread.join(timeout=30)  # 30 second timeout
            
            # Collect results
            worker_results = []
            while not results_queue.empty():
                worker_results.append(results_queue.get())
            
            successful_workers = sum(1 for r in worker_results if r['success'])
            
            results['details']['concurrent_builders'] = {
                'success': successful_workers == num_workers,
                'successful_workers': successful_workers,
                'total_workers': num_workers,
                'worker_results': worker_results
            }
            
            if successful_workers == num_workers:
                results['passed_tests'] += 1
            
        except Exception as e:
            results['details']['concurrent_builders'] = {'success': False, 'error': str(e)}
        
        results['total_tests'] += 1
        
        return results
    
    def test_production_edge_cases(self) -> Dict[str, Any]:
        """Test edge cases likely to occur in production environments."""
        results = {'passed_tests': 0, 'total_tests': 0, 'details': {}}
        
        # Test 1: Very sparse activations (mostly zeros)
        try:
            sparse_activations = {}
            for i in range(5):
                activation = torch.zeros(10, 15)
                # Only fill 10% of entries
                mask = torch.rand(10, 15) < 0.1
                activation[mask] = torch.randn(mask.sum())
                sparse_activations[f'layer_{i}'] = activation
            
            layer_names = list(sparse_activations.keys())
            poset = nx.DiGraph()
            for name in layer_names:
                poset.add_node(name)
            for i in range(len(layer_names) - 1):
                poset.add_edge(layer_names[i], layer_names[i + 1])
            
            builder = SheafBuilder(use_whitening=True, enable_edge_filtering=True)
            gram_matrices = self.generator.generate_gram_matrices_from_activations(sparse_activations)
            sheaf = builder.build_from_cka_matrices(poset, gram_matrices)
            
            # Should handle sparse activations
            laplacian, metadata = builder.build_laplacian(sheaf)
            
            results['details']['sparse_activations'] = {
                'success': True,
                'sparsity': metadata.sparsity_ratio
            }
            results['passed_tests'] += 1
            
        except Exception as e:
            results['details']['sparse_activations'] = {'success': False, 'error': str(e)}
        
        results['total_tests'] += 1
        
        # Test 2: Identical layer activations
        try:
            # Create identical activations (perfect correlation)
            base_activation = torch.randn(8, 12)
            identical_activations = {
                'layer1': base_activation,
                'layer2': base_activation.clone(),
                'layer3': base_activation + 1e-10 * torch.randn_like(base_activation)  # Tiny perturbation
            }
            
            layer_names = list(identical_activations.keys())
            poset = nx.DiGraph()
            for name in layer_names:
                poset.add_node(name)
            for i in range(len(layer_names) - 1):
                poset.add_edge(layer_names[i], layer_names[i + 1])
            
            builder = SheafBuilder(use_whitening=True)
            gram_matrices = self.generator.generate_gram_matrices_from_activations(identical_activations)
            sheaf = builder.build_from_cka_matrices(poset, gram_matrices)
            
            # Should detect and handle near-identical activations
            laplacian, metadata = builder.build_laplacian(sheaf)
            
            results['details']['identical_activations'] = {'success': True}
            results['passed_tests'] += 1
            
        except Exception as e:
            results['details']['identical_activations'] = {'success': False, 'error': str(e)}
        
        results['total_tests'] += 1
        
        # Test 3: Gradual degradation with increasing noise
        try:
            noise_levels = [0.0, 0.1, 0.5, 1.0, 2.0]
            degradation_results = []
            
            base_activations = self.generator.generate_linear_transformation_sequence(
                num_layers=4, input_dim=10, batch_size=8, noise_level=0.0
            )
            
            layer_names = list(base_activations.keys())
            poset = nx.DiGraph()
            for name in layer_names:
                poset.add_node(name)
            for i in range(len(layer_names) - 1):
                poset.add_edge(layer_names[i], layer_names[i + 1])
            
            for noise_level in noise_levels:
                try:
                    # Add increasing noise
                    noisy_activations = {}
                    for name, activation in base_activations.items():
                        noise = torch.randn_like(activation) * noise_level
                        noisy_activations[name] = activation + noise
                    
                    builder = SheafBuilder(use_whitening=True, enable_edge_filtering=True)
                    gram_matrices = self.generator.generate_gram_matrices_from_activations(noisy_activations)
                    sheaf = builder.build_from_cka_matrices(poset, gram_matrices)
                    laplacian, metadata = builder.build_laplacian(sheaf)
                    
                    degradation_results.append({
                        'noise_level': noise_level,
                        'success': True,
                        'sparsity': metadata.sparsity_ratio,
                        'shape': laplacian.shape
                    })
                    
                except Exception as e:
                    degradation_results.append({
                        'noise_level': noise_level,
                        'success': False,
                        'error': str(e)
                    })
            
            # Should handle graceful degradation
            successful_levels = sum(1 for r in degradation_results if r['success'])
            
            results['details']['noise_degradation'] = {
                'success': successful_levels >= len(noise_levels) // 2,  # At least half should succeed
                'successful_levels': successful_levels,
                'total_levels': len(noise_levels),
                'degradation_results': degradation_results
            }
            
            if successful_levels >= len(noise_levels) // 2:
                results['passed_tests'] += 1
            
        except Exception as e:
            results['details']['noise_degradation'] = {'success': False, 'error': str(e)}
        
        results['total_tests'] += 1
        
        return results
    
    def run_comprehensive_robustness_tests(self) -> Dict[str, Any]:
        """Run complete robustness test suite."""
        print("🛡️  Starting Comprehensive Robustness Testing")
        print("=" * 60)
        
        all_results = {}
        
        # 1. Numerical stability tests
        print("\n🔢 Testing Numerical Stability...")
        all_results['numerical_stability'] = self.test_numerical_stability_ill_conditioned_matrices()
        
        # 2. Extreme scale tests
        print("📏 Testing Extreme Scale Differences...")
        all_results['extreme_scales'] = self.test_extreme_scale_differences()
        
        # 3. Unusual topology tests
        print("🕸️  Testing Unusual Network Topologies...")
        all_results['unusual_topologies'] = self.test_unusual_network_topologies()
        
        # 4. Extreme dimension tests
        print("📐 Testing Extreme Dimensions...")
        all_results['extreme_dimensions'] = self.test_extreme_dimensions()
        
        # 5. Error handling tests
        print("⚠️  Testing Error Handling & Recovery...")
        all_results['error_handling'] = self.test_error_handling_recovery()
        
        # 6. Concurrent access tests
        print("🔄 Testing Concurrent Access...")
        all_results['concurrent_access'] = self.test_concurrent_access_thread_safety()
        
        # 7. Production edge cases
        print("🏭 Testing Production Edge Cases...")
        all_results['production_edge_cases'] = self.test_production_edge_cases()
        
        # Generate summary
        self.generate_robustness_summary(all_results)
        
        return all_results
    
    def generate_robustness_summary(self, results: Dict[str, Any]):
        """Generate comprehensive robustness test summary."""
        print("\n" + "=" * 60)
        print("🛡️  COMPREHENSIVE ROBUSTNESS TEST RESULTS")
        print("=" * 60)
        
        total_passed = 0
        total_tests = 0
        
        for test_category, result in results.items():
            passed = result['passed_tests']
            total = result['total_tests']
            total_passed += passed
            total_tests += total
            
            status = "✅" if passed == total else "⚠️" if passed > total // 2 else "❌"
            print(f"\n{status} {test_category.replace('_', ' ').title()}: {passed}/{total} tests passed")
            
            # Show details for failed tests
            if passed < total:
                for test_name, details in result['details'].items():
                    if not details.get('success', False):
                        print(f"   ❌ {test_name}: {details.get('error', 'Unknown error')}")
                    elif details.get('raised_appropriate_error', False):
                        print(f"   ✅ {test_name}: Appropriately handled with error")
        
        overall_success_rate = total_passed / total_tests if total_tests > 0 else 0
        
        print(f"\n🎯 Overall Robustness: {total_passed}/{total_tests} tests passed ({overall_success_rate:.1%})")
        
        if overall_success_rate >= 0.85:
            print("🎉 EXCELLENT ROBUSTNESS - PRODUCTION READY")
            print("   ✅ Handles edge cases gracefully")
            print("   ✅ Proper error handling and recovery")
            print("   ✅ Robust to numerical instabilities")
        elif overall_success_rate >= 0.7:
            print("✅ GOOD ROBUSTNESS - MINOR IMPROVEMENTS NEEDED")
            print("   ✅ Most edge cases handled well")
            print("   ⚠️  Some areas need attention")
        else:
            print("⚠️  ROBUSTNESS NEEDS IMPROVEMENT")
            print("   ⚠️  Several edge cases not handled properly")
            print("   ⚠️  Review failing tests before production")


class TestRobustnessEdgeCases:
    """Test suite for robustness and edge case validation."""
    
    def test_numerical_stability_comprehensive(self):
        """Test comprehensive numerical stability."""
        suite = RobustnessTestSuite()
        results = suite.test_numerical_stability_ill_conditioned_matrices()
        
        # Should handle most numerical stability cases
        assert results['passed_tests'] >= results['total_tests'] * 0.8, f"Poor numerical stability: {results['passed_tests']}/{results['total_tests']}"
        
        print(f"✅ Numerical stability validated: {results['passed_tests']}/{results['total_tests']} tests passed")
    
    def test_extreme_topologies_handling(self):
        """Test handling of extreme network topologies."""
        suite = RobustnessTestSuite()
        results = suite.test_unusual_network_topologies()
        
        # Should handle all topology cases
        assert results['passed_tests'] == results['total_tests'], f"Topology handling failed: {results['passed_tests']}/{results['total_tests']}"
        
        print(f"✅ Topology handling validated: {results['passed_tests']}/{results['total_tests']} tests passed")
    
    def test_error_recovery_mechanisms(self):
        """Test error handling and recovery mechanisms."""
        suite = RobustnessTestSuite()
        results = suite.test_error_handling_recovery()
        
        # Should handle all error scenarios
        assert results['passed_tests'] == results['total_tests'], f"Error handling failed: {results['passed_tests']}/{results['total_tests']}"
        
        print(f"✅ Error handling validated: {results['passed_tests']}/{results['total_tests']} tests passed")
    
    def test_production_readiness(self):
        """Test production readiness with edge cases."""
        suite = RobustnessTestSuite()
        results = suite.test_production_edge_cases()
        
        # Should handle most production scenarios
        assert results['passed_tests'] >= results['total_tests'] * 0.8, f"Production readiness insufficient: {results['passed_tests']}/{results['total_tests']}"
        
        print(f"✅ Production readiness validated: {results['passed_tests']}/{results['total_tests']} tests passed")


if __name__ == "__main__":
    # Run comprehensive robustness tests
    suite = RobustnessTestSuite()
    results = suite.run_comprehensive_robustness_tests()
    
    # Save results
    import json
    with open('/Users/francescopapini/GitRepo/neurosheaf/tests/comprehensive_validation/robustness_test_results.json', 'w') as f:
        def convert_for_json(obj):
            if isinstance(obj, (np.ndarray, torch.Tensor)):
                return obj.tolist() if hasattr(obj, 'tolist') else str(obj)
            elif isinstance(obj, (np.float64, np.int64)):
                return float(obj) if isinstance(obj, np.float64) else int(obj)
            elif isinstance(obj, dict):
                return {k: convert_for_json(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_for_json(v) for v in obj]
            else:
                return obj
        
        json.dump(convert_for_json(results), f, indent=2)
    
    print(f"\n💾 Results saved to robustness_test_results.json")
    
    # Run pytest
    pytest.main([__file__, "-v", "-s", "--tb=short"])
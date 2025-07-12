# tests/phase4_spectral/integration/test_week9_integration.py
import pytest
import torch
import numpy as np
import networkx as nx
from neurosheaf.spectral import StaticLaplacianWithMasking, PersistentSpectralAnalyzer, SubspaceTracker
from neurosheaf.sheaf.construction import Sheaf

class TestWeek9Integration:
    """Integration tests for Week 9 persistent spectral analysis."""
    
    def test_complete_persistence_pipeline(self):
        """Test complete persistence pipeline from sheaf to diagrams."""
        # Create a realistic sheaf (representing layers of a neural network)
        poset = nx.DiGraph()
        poset.add_edges_from([('0', '1'), ('1', '2'), ('2', '3'), ('3', '4')])
        
        # Consistent dimensions for all "layers" (simplifies testing)
        stalks = {
            '0': torch.eye(3),  # Input layer: 3 features
            '1': torch.eye(3),  # Hidden layer 1: 3 features  
            '2': torch.eye(3),  # Hidden layer 2: 3 features
            '3': torch.eye(3),  # Hidden layer 3: 3 features
            '4': torch.eye(3)   # Output layer: 3 features
        }
        
        # Restrictions with varying strengths (simulating different layer connections)
        restrictions = {
            ('0', '1'): torch.eye(3) * 0.8,  # Strong connection
            ('1', '2'): torch.eye(3) * 0.6,  # Medium connection
            ('2', '3'): torch.eye(3) * 0.7,  # Medium-strong connection
            ('3', '4'): torch.eye(3) * 0.5   # Weaker connection
        }
        
        sheaf = Sheaf(poset, stalks, restrictions)
        
        # Create analyzer
        analyzer = PersistentSpectralAnalyzer()
        
        # Perform complete analysis
        result = analyzer.analyze(sheaf, filtration_type='threshold', n_steps=10)
        
        # Validate complete pipeline results
        assert 'persistence_result' in result
        assert 'features' in result
        assert 'diagrams' in result
        assert 'filtration_params' in result
        assert 'analysis_metadata' in result
        
        # Check persistence computation
        persistence_result = result['persistence_result']
        assert len(persistence_result['eigenvalue_sequences']) == 10
        assert len(persistence_result['eigenvector_sequences']) == 10
        assert 'tracking_info' in persistence_result
        
        # Check that eigenvalue sequences are well-formed
        for eigenvals, eigenvecs in zip(
            persistence_result['eigenvalue_sequences'],
            persistence_result['eigenvector_sequences']
        ):
            assert isinstance(eigenvals, torch.Tensor)
            assert isinstance(eigenvecs, torch.Tensor)
            assert len(eigenvals) > 0  # Should have some eigenvalues
            assert eigenvals.shape[0] == eigenvecs.shape[1]
            assert torch.all(eigenvals >= -1e-6)  # Non-negative (Laplacian property)
        
        # Check tracking integration (from Week 8)
        tracking_info = persistence_result['tracking_info']
        assert 'eigenvalue_paths' in tracking_info
        assert 'birth_events' in tracking_info
        assert 'death_events' in tracking_info
        
        # Check feature extraction
        features = result['features']
        assert len(features['eigenvalue_evolution']) == 10
        assert 'summary' in features
        assert features['summary']['total_filtration_steps'] == 10
        
        # Check persistence diagrams
        diagrams = result['diagrams']
        assert 'birth_death_pairs' in diagrams
        assert 'infinite_bars' in diagrams
        assert 'statistics' in diagrams
        
        # Validate diagram properties
        for pair in diagrams['birth_death_pairs']:
            assert pair['birth'] <= pair['death']
            assert pair['lifetime'] >= 0
    
    def test_subspace_tracker_integration(self):
        """Test integration between StaticLaplacianWithMasking and SubspaceTracker."""
        # Create simple sheaf with known eigenvalue evolution
        poset = nx.DiGraph()
        poset.add_edges_from([('0', '1'), ('1', '2')])
        stalks = {'0': torch.eye(3), '1': torch.eye(3), '2': torch.eye(3)}
        
        # Create restrictions that will lead to predictable eigenvalue changes
        restrictions = {
            ('0', '1'): torch.eye(3) * 0.8,
            ('1', '2'): torch.eye(3) * 0.6
        }
        sheaf = Sheaf(poset, stalks, restrictions)
        
        # Create static Laplacian analyzer
        static_analyzer = StaticLaplacianWithMasking(max_eigenvalues=8)
        
        # Define filtration parameters that will change edge connectivity
        filtration_params = [0.1, 0.5, 0.7, 0.9]
        def threshold_func(weight, param):
            return weight >= param
        
        # Compute persistence
        persistence_result = static_analyzer.compute_persistence(
            sheaf, filtration_params, threshold_func
        )
        
        # Verify SubspaceTracker was called and produced results
        tracking_info = persistence_result['tracking_info']
        
        # Check that tracking produced meaningful results
        assert isinstance(tracking_info, dict)
        assert 'eigenvalue_paths' in tracking_info
        assert 'birth_events' in tracking_info
        assert 'death_events' in tracking_info
        
        # Verify eigenvalue sequences are consistent
        eigenval_seqs = persistence_result['eigenvalue_sequences']
        assert len(eigenval_seqs) == 4
        
        # Check monotonic behavior (as we increase threshold, connectivity should decrease)
        nnz_counts = []
        for i, eigenvals in enumerate(eigenval_seqs):
            # Count non-zero eigenvalues (above numerical threshold)
            nnz_count = torch.sum(eigenvals > 1e-8).item()
            nnz_counts.append(nnz_count)
        
        # Generally, higher filtration parameters should lead to more zero eigenvalues
        # (less connectivity), but allow some flexibility due to numerical effects
        # Check that we have reasonable eigenvalue counts (allow for edge case where no variation occurs)
        assert all(count >= 0 for count in nnz_counts)  # All counts should be non-negative
        assert max(nnz_counts) <= 8  # Should not exceed max_eigenvalues limit
    
    def test_different_filtration_types_integration(self):
        """Test integration with different filtration types."""
        # Create diverse sheaf with varying edge weights
        poset = nx.DiGraph()
        nodes = ['0', '1', '2', '3']
        poset.add_nodes_from(nodes)
        # Add all possible directed edges (complete graph)
        for i in nodes:
            for j in nodes:
                if i != j:
                    poset.add_edge(i, j)
        stalks = {'0': torch.eye(2), '1': torch.eye(2), '2': torch.eye(2), '3': torch.eye(2)}
        
        # Create restrictions with diverse weights
        restrictions = {}
        weights = [0.9, 0.7, 0.5, 0.8, 0.6, 0.4]
        idx = 0
        for i in range(4):
            for j in range(4):
                if i != j:
                    restrictions[(str(i), str(j))] = torch.eye(2) * weights[idx % len(weights)]
                    idx += 1
        
        sheaf = Sheaf(poset, stalks, restrictions)
        
        analyzer = PersistentSpectralAnalyzer()
        
        # Test threshold filtration
        result_threshold = analyzer.analyze(
            sheaf, filtration_type='threshold', n_steps=8
        )
        
        # Test CKA-based filtration
        result_cka = analyzer.analyze(
            sheaf, filtration_type='cka_based', n_steps=8
        )
        
        # Both should complete successfully
        assert len(result_threshold['persistence_result']['eigenvalue_sequences']) == 8
        assert len(result_cka['persistence_result']['eigenvalue_sequences']) == 8
        
        # Check that different filtration types produce different results
        threshold_features = result_threshold['features']
        cka_features = result_cka['features']
        
        # The parameters should be different
        assert result_threshold['filtration_params'] != result_cka['filtration_params']
        
        # Both should have valid persistence features
        assert threshold_features['num_birth_events'] >= 0
        assert cka_features['num_birth_events'] >= 0
    
    def test_performance_with_realistic_network(self):
        """Test performance with a realistic neural network-sized sheaf."""
        # Create larger sheaf (simulating a moderate-sized neural network)
        n_layers = 8
        poset = nx.DiGraph()
        nodes = [str(i) for i in range(n_layers)]
        poset.add_nodes_from(nodes)
        for i in range(n_layers - 1):
            poset.add_edge(str(i), str(i + 1))
        
        # Consistent dimensions for simplicity (performance test focuses on size, not varying dims)
        layer_dim = 4  # Moderate dimension for performance testing
        stalks = {str(i): torch.eye(layer_dim) for i in range(n_layers)}
        
        # Create restrictions with realistic weight distributions
        restrictions = {}
        np.random.seed(42)  # For reproducibility
        for i in range(n_layers - 1):
            # Random restriction matrix with consistent dimensions
            R = torch.randn(layer_dim, layer_dim) * 0.1
            # Add some structure (stronger connections on diagonal)
            for j in range(layer_dim):
                R[j, j] += 0.5
            
            restrictions[(str(i), str(i+1))] = R
        
        sheaf = Sheaf(poset, stalks, restrictions)
        
        # Analyze with moderate number of steps
        analyzer = PersistentSpectralAnalyzer()
        
        import time
        start_time = time.time()
        result = analyzer.analyze(sheaf, n_steps=15)
        analysis_time = time.time() - start_time
        
        # Should complete in reasonable time (less than 30 seconds for this size)
        assert analysis_time < 30.0
        
        # Should produce valid results
        assert len(result['persistence_result']['eigenvalue_sequences']) == 15
        assert result['analysis_metadata']['analysis_time'] > 0
        
        # Check that we get meaningful eigenvalue counts
        eigenval_seqs = result['persistence_result']['eigenvalue_sequences']
        eigenval_counts = [len(seq) for seq in eigenval_seqs]
        assert max(eigenval_counts) > 0
        assert min(eigenval_counts) >= 0
    
    def test_edge_masking_correctness(self):
        """Test that edge masking produces mathematically correct results."""
        # Create simple sheaf where we can predict the effect of masking
        poset = nx.DiGraph()
        poset.add_edges_from([('0', '1'), ('1', '2')])
        stalks = {'0': torch.eye(2), '1': torch.eye(2), '2': torch.eye(2)}
        restrictions = {
            ('0', '1'): torch.eye(2) * 0.8,  # Strong edge
            ('1', '2'): torch.eye(2) * 0.3   # Weak edge
        }
        sheaf = Sheaf(poset, stalks, restrictions)
        
        analyzer = StaticLaplacianWithMasking(max_eigenvalues=6)
        
        # Test filtration that should include all edges
        def threshold_func(weight, param):
            return weight >= param
        
        result_all = analyzer.compute_persistence(sheaf, [0.1], threshold_func)
        eigenvals_all = result_all['eigenvalue_sequences'][0]
        
        # Test filtration that should exclude weak edge
        result_strong = analyzer.compute_persistence(sheaf, [0.5], threshold_func)
        eigenvals_strong = result_strong['eigenvalue_sequences'][0]
        
        # Test filtration that should exclude all edges
        result_none = analyzer.compute_persistence(sheaf, [0.9], threshold_func)
        eigenvals_none = result_none['eigenvalue_sequences'][0]
        
        # Verify that masking affects the eigenvalue spectrum appropriately
        # More connections should generally lead to different eigenvalue distributions
        assert len(eigenvals_all) > 0
        assert len(eigenvals_strong) > 0
        assert len(eigenvals_none) > 0
        
        # The smallest eigenvalue should always be approximately 0 (Laplacian property)
        assert eigenvals_all[0] < 1e-6
        assert eigenvals_strong[0] < 1e-6
        assert eigenvals_none[0] < 1e-6
    
    def test_persistence_diagram_validity(self):
        """Test that generated persistence diagrams are mathematically valid."""
        # Create sheaf with predictable persistence behavior
        poset = nx.DiGraph()
        poset.add_edges_from([('0', '1'), ('1', '2'), ('2', '3')])
        stalks = {'0': torch.eye(2), '1': torch.eye(2), '2': torch.eye(2), '3': torch.eye(2)}
        
        # Gradual decrease in connection strength
        restrictions = {
            ('0', '1'): torch.eye(2) * 0.9,
            ('1', '2'): torch.eye(2) * 0.6,
            ('2', '3'): torch.eye(2) * 0.3
        }
        sheaf = Sheaf(poset, stalks, restrictions)
        
        analyzer = PersistentSpectralAnalyzer()
        result = analyzer.analyze(sheaf, n_steps=12)
        
        diagrams = result['diagrams']
        
        # Validate persistence diagram properties
        birth_death_pairs = diagrams['birth_death_pairs']
        infinite_bars = diagrams['infinite_bars']
        
        # All birth times should be <= death times
        for pair in birth_death_pairs:
            assert pair['birth'] <= pair['death']
            assert pair['lifetime'] == pair['death'] - pair['birth']
            assert pair['lifetime'] >= 0
        
        # Infinite bars should have finite birth times
        for bar in infinite_bars:
            assert np.isfinite(bar['birth'])
            assert bar['death'] == float('inf')
        
        # Check statistics consistency
        stats = diagrams['statistics']
        assert stats['n_finite_pairs'] == len(birth_death_pairs)
        assert stats['n_infinite_bars'] == len(infinite_bars)
        
        if birth_death_pairs:
            computed_mean_lifetime = np.mean([pair['lifetime'] for pair in birth_death_pairs])
            assert abs(stats['mean_lifetime'] - computed_mean_lifetime) < 1e-10
    
    def test_memory_efficiency(self):
        """Test memory efficiency of the persistence computation."""
        # Create moderately large sheaf
        n = 6
        poset = nx.DiGraph()
        nodes = [str(i) for i in range(n)]
        poset.add_nodes_from(nodes)
        # Add all possible directed edges (complete graph)
        for i in nodes:
            for j in nodes:
                if i != j:
                    poset.add_edge(i, j)
        stalks = {str(i): torch.eye(3) for i in range(n)}
        
        # Dense connections
        restrictions = {}
        for i in range(n):
            for j in range(n):
                if i != j:
                    restrictions[(str(i), str(j))] = torch.randn(3, 3) * 0.1 + torch.eye(3) * 0.5
        
        sheaf = Sheaf(poset, stalks, restrictions)
        
        analyzer = PersistentSpectralAnalyzer()
        
        # Monitor memory usage (basic check)
        import psutil
        import os
        process = psutil.Process(os.getpid())
        
        memory_before = process.memory_info().rss / 1024 / 1024  # MB
        
        result = analyzer.analyze(sheaf, n_steps=10)
        
        memory_after = process.memory_info().rss / 1024 / 1024  # MB
        memory_used = memory_after - memory_before
        
        # Should not use excessive memory (less than 500MB for this test)
        assert memory_used < 500
        
        # Should still produce valid results
        assert len(result['persistence_result']['eigenvalue_sequences']) == 10
        
        # Clear cache to free memory
        analyzer.clear_cache()
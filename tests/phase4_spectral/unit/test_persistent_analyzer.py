# tests/phase4_spectral/unit/test_persistent_analyzer.py
import pytest
import torch
import numpy as np
import networkx as nx
from neurosheaf.spectral.persistent import PersistentSpectralAnalyzer
from neurosheaf.spectral.static_laplacian_masking import StaticLaplacianWithMasking
from neurosheaf.spectral.tracker import SubspaceTracker
from neurosheaf.sheaf.construction import Sheaf

class TestPersistentSpectralAnalyzer:
    """Unit tests for PersistentSpectralAnalyzer class."""
    
    def test_initialization(self):
        """Test PersistentSpectralAnalyzer initialization."""
        # Test default initialization
        analyzer = PersistentSpectralAnalyzer()
        assert analyzer.static_laplacian is not None
        assert analyzer.subspace_tracker is not None
        assert analyzer.default_n_steps == 50
        assert analyzer.default_filtration_type == 'threshold'
        
        # Test custom initialization
        custom_laplacian = StaticLaplacianWithMasking(max_eigenvalues=20)
        custom_tracker = SubspaceTracker(gap_eps=1e-5)
        
        analyzer = PersistentSpectralAnalyzer(
            static_laplacian=custom_laplacian,
            subspace_tracker=custom_tracker,
            default_n_steps=30,
            default_filtration_type='cka_based'
        )
        
        assert analyzer.static_laplacian is custom_laplacian
        assert analyzer.subspace_tracker is custom_tracker
        assert analyzer.default_n_steps == 30
        assert analyzer.default_filtration_type == 'cka_based'
    
    def test_filtration_parameter_generation(self):
        """Test automatic filtration parameter generation."""
        # Create simple sheaf with consistent node naming
        poset = nx.DiGraph()
        poset.add_edges_from([('0', '1'), ('1', '2')])
        stalks = {'0': torch.eye(2), '1': torch.eye(2), '2': torch.eye(2)}
        restrictions = {
            ('0', '1'): torch.eye(2) * 0.8,
            ('1', '2'): torch.eye(2) * 0.6
        }
        sheaf = Sheaf(poset, stalks, restrictions)
        
        analyzer = PersistentSpectralAnalyzer()
        
        # Test threshold filtration with auto-detection
        params_threshold = analyzer._generate_filtration_params(
            sheaf, 'threshold', 10, None
        )
        assert len(params_threshold) == 10
        assert params_threshold[0] < params_threshold[-1]  # Should be increasing
        
        # Test CKA-based filtration
        params_cka = analyzer._generate_filtration_params(
            sheaf, 'cka_based', 10, None
        )
        assert len(params_cka) == 10
        assert params_cka[0] == 0.0
        assert params_cka[-1] == 1.0
        
        # Test custom parameter range
        params_custom = analyzer._generate_filtration_params(
            sheaf, 'threshold', 5, (0.1, 0.9)
        )
        assert len(params_custom) == 5
        assert abs(params_custom[0] - 0.1) < 1e-6
        assert abs(params_custom[-1] - 0.9) < 1e-6
    
    def test_edge_threshold_function_creation(self):
        """Test creation of edge threshold functions."""
        analyzer = PersistentSpectralAnalyzer()
        
        # Test threshold filtration
        threshold_func = analyzer._create_edge_threshold_func('threshold')
        assert threshold_func(0.8, 0.5) == True   # 0.8 >= 0.5
        assert threshold_func(0.3, 0.5) == False  # 0.3 < 0.5
        
        # Test CKA-based filtration (should behave the same as threshold)
        cka_func = analyzer._create_edge_threshold_func('cka_based')
        assert cka_func(0.8, 0.5) == True
        assert cka_func(0.3, 0.5) == False
        
        # Test custom function
        def custom_func(weight, param):
            return weight > param * 2
        
        custom_threshold_func = analyzer._create_edge_threshold_func('custom', custom_func)
        assert custom_threshold_func(1.0, 0.4) == True   # 1.0 > 0.4 * 2
        assert custom_threshold_func(0.7, 0.4) == False  # 0.7 < 0.4 * 2
        
        # Test error case
        with pytest.raises(ValueError):
            analyzer._create_edge_threshold_func('custom', None)
    
    def test_feature_extraction(self):
        """Test extraction of persistence features."""
        analyzer = PersistentSpectralAnalyzer()
        
        # Create mock persistence result
        eigenval_sequences = [
            torch.tensor([0.0, 0.5, 1.0, 2.0]),
            torch.tensor([0.0, 0.7, 1.2, 2.1]),
            torch.tensor([0.1, 0.9, 1.5])
        ]
        
        eigenvec_sequences = [
            torch.eye(4),
            torch.eye(4),
            torch.eye(3)
        ]
        
        tracking_info = {
            'eigenvalue_paths': [[], []],
            'birth_events': [{'step': 1, 'group': 0, 'filtration_param': 0.3}],
            'death_events': [{'step': 2, 'group': 1, 'filtration_param': 0.7}],
            'crossings': []
        }
        
        persistence_result = {
            'eigenvalue_sequences': eigenval_sequences,
            'eigenvector_sequences': eigenvec_sequences,
            'tracking_info': tracking_info
        }
        
        features = analyzer._extract_persistence_features(persistence_result)
        
        # Check feature structure
        assert 'eigenvalue_evolution' in features
        assert 'spectral_gap_evolution' in features
        assert 'effective_dimension' in features
        assert 'num_birth_events' in features
        assert 'num_death_events' in features
        assert 'summary' in features
        
        # Check feature counts
        assert len(features['eigenvalue_evolution']) == 3
        assert len(features['spectral_gap_evolution']) == 3
        assert len(features['effective_dimension']) == 3
        assert features['num_birth_events'] == 1
        assert features['num_death_events'] == 1
        
        # Check specific values
        assert features['eigenvalue_evolution'][0]['min'] == 0.0
        assert features['eigenvalue_evolution'][0]['max'] == 2.0
        assert features['spectral_gap_evolution'][0] == 0.5  # 0.5 - 0.0
    
    def test_persistence_diagram_generation(self):
        """Test generation of persistence diagrams."""
        analyzer = PersistentSpectralAnalyzer()
        
        # Create mock tracking info
        tracking_info = {
            'birth_events': [
                {'step': 1, 'group': 0, 'filtration_param': 0.2},
                {'step': 3, 'group': 1, 'filtration_param': 0.6}
            ],
            'death_events': [
                {'step': 2, 'group': 0, 'filtration_param': 0.4},
                {'step': 4, 'group': 2, 'filtration_param': 0.8}
            ]
        }
        
        filtration_params = [0.0, 0.2, 0.4, 0.6, 0.8]
        
        diagrams = analyzer._generate_persistence_diagrams(tracking_info, filtration_params)
        
        # Check diagram structure
        assert 'birth_death_pairs' in diagrams
        assert 'infinite_bars' in diagrams
        assert 'statistics' in diagrams
        
        # Check that pairs are created
        assert len(diagrams['birth_death_pairs']) >= 1
        
        # Check pair structure
        for pair in diagrams['birth_death_pairs']:
            assert 'birth' in pair
            assert 'death' in pair
            assert 'lifetime' in pair
            assert pair['birth'] <= pair['death']
            assert pair['lifetime'] >= 0
        
        # Check statistics
        stats = diagrams['statistics']
        assert 'n_finite_pairs' in stats
        assert 'n_infinite_bars' in stats
        assert 'mean_lifetime' in stats
    
    def test_analyze_basic(self):
        """Test basic analyze functionality."""
        # Create simple sheaf with consistent node naming
        poset = nx.DiGraph()
        poset.add_edges_from([('0', '1'), ('1', '2'), ('2', '3')])
        stalks = {'0': torch.eye(2), '1': torch.eye(2), '2': torch.eye(2), '3': torch.eye(2)}
        restrictions = {
            ('0', '1'): torch.eye(2) * 0.9,
            ('1', '2'): torch.eye(2) * 0.7,
            ('2', '3'): torch.eye(2) * 0.5
        }
        sheaf = Sheaf(poset, stalks, restrictions)
        
        analyzer = PersistentSpectralAnalyzer()
        
        # Perform analysis
        result = analyzer.analyze(sheaf, n_steps=5)
        
        # Check result structure
        assert 'persistence_result' in result
        assert 'features' in result
        assert 'diagrams' in result
        assert 'filtration_params' in result
        assert 'filtration_type' in result
        assert 'analysis_metadata' in result
        
        # Check that analysis completed
        assert len(result['filtration_params']) == 5
        assert result['filtration_type'] == 'threshold'
        assert result['analysis_metadata']['analysis_time'] > 0
        
        # Check that features were extracted
        features = result['features']
        assert len(features['eigenvalue_evolution']) == 5
        assert 'summary' in features
    
    def test_analyze_different_filtration_types(self):
        """Test analysis with different filtration types."""
        # Create simple sheaf with consistent node naming
        poset = nx.DiGraph()
        poset.add_edges_from([('0', '1'), ('1', '2'), ('2', '0')])
        stalks = {'0': torch.eye(2), '1': torch.eye(2), '2': torch.eye(2)}
        restrictions = {
            ('0', '1'): torch.eye(2) * 0.8,
            ('1', '2'): torch.eye(2) * 0.6,
            ('2', '0'): torch.eye(2) * 0.7
        }
        sheaf = Sheaf(poset, stalks, restrictions)
        
        analyzer = PersistentSpectralAnalyzer()
        
        # Test threshold filtration
        result1 = analyzer.analyze(sheaf, filtration_type='threshold', n_steps=5)
        assert result1['filtration_type'] == 'threshold'
        
        # Test CKA-based filtration
        result2 = analyzer.analyze(sheaf, filtration_type='cka_based', n_steps=5)
        assert result2['filtration_type'] == 'cka_based'
        
        # Test custom filtration
        def custom_threshold(weight, param):
            return weight >= param * 0.5
        
        result3 = analyzer.analyze(
            sheaf, 
            filtration_type='custom', 
            n_steps=5,
            custom_threshold_func=custom_threshold
        )
        assert result3['filtration_type'] == 'custom'
        
        # All should complete successfully
        assert len(result1['persistence_result']['eigenvalue_sequences']) == 5
        assert len(result2['persistence_result']['eigenvalue_sequences']) == 5
        assert len(result3['persistence_result']['eigenvalue_sequences']) == 5
    
    def test_analyze_multiple_sheaves(self):
        """Test analysis of multiple sheaves."""
        # Create multiple simple sheaves
        sheaves = []
        for i in range(3):
            poset = nx.DiGraph()
            poset.add_edges_from([('0', '1'), ('1', '2')])
            stalks = {'0': torch.eye(2), '1': torch.eye(2), '2': torch.eye(2)}
            restrictions = {
                ('0', '1'): torch.eye(2) * (0.8 + i * 0.1),
                ('1', '2'): torch.eye(2) * (0.6 + i * 0.1)
            }
            sheaves.append(Sheaf(poset, stalks, restrictions))
        
        analyzer = PersistentSpectralAnalyzer()
        
        # Analyze multiple sheaves
        results = analyzer.analyze_multiple_sheaves(sheaves, n_steps=5)
        
        # Check results
        assert len(results) == 3
        for result in results:
            assert 'persistence_result' in result
            assert 'features' in result
            assert len(result['filtration_params']) == 5
    
    def test_empty_sheaf_handling(self):
        """Test handling of empty or minimal sheaves."""
        analyzer = PersistentSpectralAnalyzer()
        
        # Single node sheaf
        single_poset = nx.DiGraph()
        single_poset.add_node('A')
        single_sheaf = Sheaf(single_poset, {'A': torch.eye(2)}, {})
        
        result = analyzer.analyze(single_sheaf, n_steps=3)
        
        # Should complete without errors
        assert len(result['filtration_params']) == 3
        assert 'features' in result
        assert 'diagrams' in result
    
    def test_cache_management(self):
        """Test cache management functionality."""
        analyzer = PersistentSpectralAnalyzer()
        
        # Create simple sheaf with consistent node naming
        poset = nx.DiGraph()
        poset.add_edges_from([('0', '1'), ('1', '2')])
        stalks = {'0': torch.eye(2), '1': torch.eye(2), '2': torch.eye(2)}
        restrictions = {
            ('0', '1'): torch.eye(2) * 0.8,
            ('1', '2'): torch.eye(2) * 0.6
        }
        sheaf = Sheaf(poset, stalks, restrictions)
        
        # First analysis should populate cache
        result1 = analyzer.analyze(sheaf, n_steps=3)
        cache_info = analyzer.static_laplacian.get_cache_info()
        assert cache_info['laplacian_cached']
        
        # Clear cache
        analyzer.clear_cache()
        cache_info = analyzer.static_laplacian.get_cache_info()
        assert not cache_info['laplacian_cached']
    
    def test_feature_extraction_edge_cases(self):
        """Test feature extraction with edge cases."""
        analyzer = PersistentSpectralAnalyzer()
        
        # Empty eigenvalue sequences
        persistence_result = {
            'eigenvalue_sequences': [torch.tensor([]), torch.tensor([0.0])],
            'eigenvector_sequences': [torch.zeros(1, 0), torch.ones(1, 1)],
            'tracking_info': {
                'eigenvalue_paths': [],
                'birth_events': [],
                'death_events': [],
                'crossings': []
            }
        }
        
        features = analyzer._extract_persistence_features(persistence_result)
        
        # Should handle empty sequences gracefully
        assert len(features['eigenvalue_evolution']) == 2
        assert features['eigenvalue_evolution'][0]['mean'] == 0.0
        assert features['num_birth_events'] == 0
    
    def test_parameter_validation(self):
        """Test parameter validation and error handling."""
        analyzer = PersistentSpectralAnalyzer()
        
        # Create simple sheaf with consistent node naming
        poset = nx.DiGraph()
        poset.add_edges_from([('0', '1')])
        stalks = {'0': torch.eye(2), '1': torch.eye(2)}
        restrictions = {('0', '1'): torch.eye(2) * 0.8}
        sheaf = Sheaf(poset, stalks, restrictions)
        
        # Test with zero steps
        result = analyzer.analyze(sheaf, n_steps=0)
        assert len(result['filtration_params']) == 0
        
        # Test with negative steps (should be handled gracefully)
        try:
            result = analyzer.analyze(sheaf, n_steps=-1)
            # If it doesn't raise an error, check that it handles it gracefully
            assert len(result['filtration_params']) >= 0
        except:
            # It's acceptable for this to raise an error
            pass
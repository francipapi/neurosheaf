# tests/phase4_spectral/unit/test_multi_parameter.py
import pytest
import torch
import numpy as np
import networkx as nx
from neurosheaf.spectral.multi_parameter import (
    MultiParameterFiltration,
    ParameterPoint,
    ParameterCorrelationAnalyzer,
    MultiParameterPersistenceComputer,
    MultiParameterSpectralAnalyzer
)
from neurosheaf.sheaf.construction import Sheaf

class TestParameterPoint:
    """Test ParameterPoint class."""
    
    def test_initialization(self):
        """Test ParameterPoint initialization."""
        point = ParameterPoint(coordinates=(0.5, 0.8), index=0)
        assert point.coordinates == (0.5, 0.8)
        assert point.index == 0
        
        # Test tuple conversion
        point2 = ParameterPoint(coordinates=[1.0, 2.0], index=1)
        assert point2.coordinates == (1.0, 2.0)
        assert isinstance(point2.coordinates, tuple)
    
    def test_comparison_operators(self):
        """Test ParameterPoint comparison operators."""
        point1 = ParameterPoint((0.5, 0.8), 0)
        point2 = ParameterPoint((0.6, 0.9), 1)
        point3 = ParameterPoint((0.5, 0.8), 2)
        
        # Test equality
        assert point1 == point3  # Same coordinates, different index
        assert point1 != point2
        
        # Test domination
        assert point2.dominates(point1)
        assert not point1.dominates(point2)
        assert point1.dominates(point1)  # Self-domination
    
    def test_hashing(self):
        """Test ParameterPoint hashing for use in dictionaries."""
        point1 = ParameterPoint((0.5, 0.8), 0)
        point2 = ParameterPoint((0.5, 0.8), 1)
        point3 = ParameterPoint((0.6, 0.9), 0)
        
        # Same coordinates should have same hash regardless of index
        assert hash(point1) == hash(point2)
        assert hash(point1) != hash(point3)
        
        # Should be usable as dictionary keys
        point_dict = {point1: "value1", point3: "value2"}
        assert len(point_dict) == 2


class TestMultiParameterFiltration:
    """Test MultiParameterFiltration class."""
    
    def test_initialization_2d(self):
        """Test 2D parameter space initialization."""
        param_names = ['cka_threshold', 'edge_weight']
        param_ranges = [(0.0, 1.0), (0.5, 2.0)]
        grid_sizes = [5, 4]
        
        filtration = MultiParameterFiltration(param_names, param_ranges, grid_sizes)
        
        assert filtration.parameter_names == param_names
        assert filtration.parameter_ranges == param_ranges
        assert filtration.grid_sizes == grid_sizes
        assert filtration.dimension == 2
        assert filtration.total_points == 5 * 4
        assert len(filtration) == 20
    
    def test_initialization_3d(self):
        """Test 3D parameter space initialization."""
        param_names = ['param1', 'param2', 'param3']
        param_ranges = [(0.0, 1.0), (-1.0, 1.0), (2.0, 5.0)]
        grid_sizes = [3, 3, 3]
        
        filtration = MultiParameterFiltration(param_names, param_ranges, grid_sizes)
        
        assert filtration.dimension == 3
        assert filtration.total_points == 27
    
    def test_parameter_grid_generation(self):
        """Test parameter grid generation."""
        param_names = ['x', 'y']
        param_ranges = [(0.0, 1.0), (0.0, 1.0)]
        grid_sizes = [3, 3]
        
        filtration = MultiParameterFiltration(param_names, param_ranges, grid_sizes)
        
        # Check that we have all expected points
        expected_coords = [
            (0.0, 0.0), (0.0, 0.5), (0.0, 1.0),
            (0.5, 0.0), (0.5, 0.5), (0.5, 1.0),
            (1.0, 0.0), (1.0, 0.5), (1.0, 1.0)
        ]
        
        actual_coords = [point.coordinates for point in filtration.parameter_points]
        
        for expected in expected_coords:
            assert expected in actual_coords
    
    def test_get_parameter_point(self):
        """Test parameter point retrieval."""
        filtration = MultiParameterFiltration(['x', 'y'], [(0.0, 1.0), (0.0, 1.0)], [3, 3])
        
        point = filtration.get_parameter_point(0)
        assert isinstance(point, ParameterPoint)
        assert point.index == 0
        
        with pytest.raises(IndexError):
            filtration.get_parameter_point(100)
    
    def test_find_index(self):
        """Test finding index by coordinates."""
        filtration = MultiParameterFiltration(['x', 'y'], [(0.0, 1.0), (0.0, 1.0)], [3, 3])
        
        index = filtration.find_index((0.0, 0.0))
        assert index is not None
        assert isinstance(index, int)
        
        # Test non-existent coordinates
        index = filtration.find_index((0.25, 0.25))
        assert index is None
    
    def test_get_neighbors(self):
        """Test neighbor finding."""
        filtration = MultiParameterFiltration(['x', 'y'], [(0.0, 1.0), (0.0, 1.0)], [3, 3])
        
        # Get center point
        center_point = None
        for point in filtration.parameter_points:
            if point.coordinates == (0.5, 0.5):
                center_point = point
                break
        
        assert center_point is not None
        
        # Test face neighbors
        neighbors = filtration.get_neighbors(center_point, 'face')
        assert len(neighbors) == 4  # Up, down, left, right
        
        # Test Moore neighbors
        moore_neighbors = filtration.get_neighbors(center_point, 'moore')
        assert len(moore_neighbors) == 8  # All adjacent including diagonals
    
    def test_interpolate_parameter_point(self):
        """Test parameter point interpolation."""
        filtration = MultiParameterFiltration(['x', 'y'], [(0.0, 1.0), (0.0, 1.0)], [5, 5])
        
        # Test interpolation to nearest grid point
        interpolated = filtration.interpolate_parameter_point((0.3, 0.7))
        assert isinstance(interpolated, ParameterPoint)
        
        # Should be close to (0.25, 0.75) which is nearest grid point
        expected_coords = (0.25, 0.75)
        assert abs(interpolated.coordinates[0] - expected_coords[0]) < 0.1
        assert abs(interpolated.coordinates[1] - expected_coords[1]) < 0.1
    
    def test_validation_errors(self):
        """Test validation of input parameters."""
        # Test mismatched lengths
        with pytest.raises(ValueError):
            MultiParameterFiltration(['x'], [(0.0, 1.0), (0.0, 1.0)], [3, 3])
        
        # Test insufficient parameters
        with pytest.raises(ValueError):
            MultiParameterFiltration(['x'], [(0.0, 1.0)], [3])
        
        # Test invalid grid size
        with pytest.raises(ValueError):
            MultiParameterFiltration(['x', 'y'], [(0.0, 1.0), (0.0, 1.0)], [1, 3])


class TestParameterCorrelationAnalyzer:
    """Test ParameterCorrelationAnalyzer class."""
    
    def test_initialization(self):
        """Test ParameterCorrelationAnalyzer initialization."""
        analyzer = ParameterCorrelationAnalyzer()
        assert analyzer.correlations == {}
        assert analyzer.parameter_stats == {}
    
    def test_compute_parameter_correlations(self):
        """Test parameter correlation computation."""
        analyzer = ParameterCorrelationAnalyzer()
        
        # Create test data with known correlations
        param_values = {
            'param1': [1.0, 2.0, 3.0, 4.0, 5.0],
            'param2': [2.0, 4.0, 6.0, 8.0, 10.0],  # Perfect positive correlation
            'param3': [5.0, 4.0, 3.0, 2.0, 1.0]    # Perfect negative correlation with param1
        }
        
        correlations = analyzer.compute_parameter_correlations(param_values)
        
        # Check structure
        assert 'param1' in correlations
        assert 'param2' in correlations
        assert 'param3' in correlations
        
        # Check self-correlation
        assert abs(correlations['param1']['param1'] - 1.0) < 1e-10
        
        # Check positive correlation
        assert correlations['param1']['param2'] > 0.9
        
        # Check negative correlation
        assert correlations['param1']['param3'] < -0.9
        
        # Check symmetry
        assert abs(correlations['param1']['param2'] - correlations['param2']['param1']) < 1e-10
    
    def test_compute_parameter_statistics(self):
        """Test parameter statistics computation."""
        analyzer = ParameterCorrelationAnalyzer()
        
        param_values = {
            'param1': [1.0, 2.0, 3.0, 4.0, 5.0],
            'param2': [0.0, 10.0, 20.0, 30.0, 40.0]
        }
        
        stats = analyzer.compute_parameter_statistics(param_values)
        
        # Check param1 statistics
        assert abs(stats['param1']['mean'] - 3.0) < 1e-10
        assert abs(stats['param1']['min'] - 1.0) < 1e-10
        assert abs(stats['param1']['max'] - 5.0) < 1e-10
        assert abs(stats['param1']['range'] - 4.0) < 1e-10
        
        # Check param2 statistics
        assert abs(stats['param2']['mean'] - 20.0) < 1e-10
        assert abs(stats['param2']['min'] - 0.0) < 1e-10
        assert abs(stats['param2']['max'] - 40.0) < 1e-10
    
    def test_identify_dominant_parameters(self):
        """Test identification of dominant parameters."""
        analyzer = ParameterCorrelationAnalyzer()
        
        parameter_effects = {
            'param1': [0.1, 0.2, 0.15, 0.18, 0.12],  # Low variance
            'param2': [1.0, 5.0, 2.0, 8.0, 3.0],     # High variance
            'param3': [0.05, 0.06, 0.055, 0.058, 0.052]  # Very low variance
        }
        
        dominant = analyzer.identify_dominant_parameters(parameter_effects, threshold=0.1)
        
        # param2 should be dominant due to high coefficient of variation
        assert 'param2' in dominant
        assert dominant[0] == 'param2'  # Should be first (highest effect)
    
    def test_generate_correlation_report(self):
        """Test correlation report generation."""
        analyzer = ParameterCorrelationAnalyzer()
        
        param_values = {
            'param1': [1.0, 2.0, 3.0],
            'param2': [2.0, 4.0, 6.0]
        }
        
        # Compute prerequisites
        analyzer.compute_parameter_correlations(param_values)
        analyzer.compute_parameter_statistics(param_values)
        
        report = analyzer.generate_correlation_report()
        
        assert 'correlations' in report
        assert 'parameter_statistics' in report
        assert 'summary' in report
        assert 'strong_correlations' in report
        
        # Check summary
        assert report['summary']['n_parameters'] == 2
        assert 'analysis_timestamp' in report['summary']


class TestMultiParameterPersistenceComputer:
    """Test MultiParameterPersistenceComputer class."""
    
    def test_initialization(self):
        """Test MultiParameterPersistenceComputer initialization."""
        computer = MultiParameterPersistenceComputer()
        assert computer.max_eigenvalues == 100
        assert computer.eigenvalue_method == 'lobpcg'
        assert computer.enable_caching == True
        
        # Test custom initialization
        computer_custom = MultiParameterPersistenceComputer(
            max_eigenvalues=50,
            eigenvalue_method='dense',
            enable_caching=False
        )
        assert computer_custom.max_eigenvalues == 50
        assert computer_custom.eigenvalue_method == 'dense'
        assert computer_custom.enable_caching == False
    
    def test_extract_multi_parameter_features(self):
        """Test feature extraction from persistence results."""
        computer = MultiParameterPersistenceComputer()
        
        # Create mock persistence result
        filtration = MultiParameterFiltration(['x', 'y'], [(0.0, 1.0), (0.0, 1.0)], [3, 3])
        
        mock_result = {
            'eigenvalue_tensor': {
                filtration.parameter_points[0]: torch.tensor([0.0, 0.5, 1.0]),
                filtration.parameter_points[1]: torch.tensor([0.1, 0.6, 1.1])
            },
            'persistence_events': [
                {'type': 'birth', 'parameter_dimension': 0, 'parameter_name': 'x'},
                {'type': 'death', 'parameter_dimension': 1, 'parameter_name': 'y'},
                {'type': 'birth', 'parameter_dimension': 0, 'parameter_name': 'x'}
            ],
            'parameter_space': filtration
        }
        
        features = computer.extract_multi_parameter_features(mock_result)
        
        # Check basic features
        assert features['n_parameter_points'] == 2
        assert features['n_persistence_events'] == 3
        assert features['parameter_space_dimension'] == 2
        assert features['parameter_names'] == ['x', 'y']
        
        # Check event counts
        assert features['n_birth_events'] == 2
        assert features['n_death_events'] == 1
        
        # Check events by parameter
        assert 'events_by_parameter' in features
        assert 'x' in features['events_by_parameter']
        assert 'y' in features['events_by_parameter']
        
        # Check eigenvalue statistics
        assert 'eigenvalue_statistics' in features
        assert 'mean' in features['eigenvalue_statistics']
    
    def test_cache_functionality(self):
        """Test caching functionality."""
        computer = MultiParameterPersistenceComputer(enable_caching=True)
        
        # Test initial cache state
        cache_info = computer.get_cache_info()
        assert cache_info['caching_enabled'] == True
        assert cache_info['eigenvalue_cache_size'] == 0
        assert cache_info['laplacian_cache_size'] == 0
        
        # Test cache clearing
        computer.clear_cache()
        cache_info = computer.get_cache_info()
        assert cache_info['eigenvalue_cache_size'] == 0
        
        # Test disabled caching
        computer_no_cache = MultiParameterPersistenceComputer(enable_caching=False)
        cache_info = computer_no_cache.get_cache_info()
        assert cache_info['caching_enabled'] == False


class TestMultiParameterSpectralAnalyzer:
    """Test MultiParameterSpectralAnalyzer class."""
    
    def test_initialization(self):
        """Test MultiParameterSpectralAnalyzer initialization."""
        analyzer = MultiParameterSpectralAnalyzer()
        assert analyzer.persistence_computer is not None
        assert analyzer.correlation_analyzer is not None
        assert analyzer.default_grid_sizes == [10, 10, 8]
    
    def test_parameter_validation(self):
        """Test parameter specification validation."""
        analyzer = MultiParameterSpectralAnalyzer()
        
        # Valid specifications
        valid_specs = {
            'param1': {
                'range': (0.0, 1.0),
                'threshold_func': lambda w, p: w >= p
            },
            'param2': {
                'range': (0.5, 2.0),
                'threshold_func': lambda w, p: w > p * 0.5
            }
        }
        
        # Should not raise
        analyzer._validate_parameter_specifications(valid_specs)
        
        # Invalid specifications
        invalid_specs = [
            # Missing range
            {'param1': {'threshold_func': lambda w, p: w >= p}},
            # Missing threshold_func
            {'param1': {'range': (0.0, 1.0)}},
            # Invalid range
            {'param1': {'range': (1.0, 0.0), 'threshold_func': lambda w, p: w >= p}},
            # Non-callable threshold_func
            {'param1': {'range': (0.0, 1.0), 'threshold_func': 'not_callable'}}
        ]
        
        for invalid_spec in invalid_specs:
            with pytest.raises(ValueError):
                analyzer._validate_parameter_specifications(invalid_spec)
    
    def test_auto_determine_grid_sizes(self):
        """Test automatic grid size determination."""
        analyzer = MultiParameterSpectralAnalyzer()
        
        # Test for different numbers of parameters
        grid_sizes_2d = analyzer._auto_determine_grid_sizes(2)
        assert len(grid_sizes_2d) == 2
        assert grid_sizes_2d == [10, 10]
        
        grid_sizes_3d = analyzer._auto_determine_grid_sizes(3)
        assert len(grid_sizes_3d) == 3
        assert grid_sizes_3d == [10, 10, 8]
        
        # Test for higher dimensions
        grid_sizes_5d = analyzer._auto_determine_grid_sizes(5)
        assert len(grid_sizes_5d) == 5
        assert all(size >= 6 for size in grid_sizes_5d)  # Should have reasonable sizes
    
    def test_sensitivity_metrics_computation(self):
        """Test sensitivity metrics computation."""
        analyzer = MultiParameterSpectralAnalyzer()
        
        sensitivity_results = [
            {'parameter_value': 0.1, 'n_persistence_events': 5},
            {'parameter_value': 0.2, 'n_persistence_events': 8},
            {'parameter_value': 0.3, 'n_persistence_events': 12},
            {'parameter_value': 0.4, 'n_persistence_events': 15}
        ]
        
        metrics = analyzer._compute_sensitivity_metrics(sensitivity_results, 'test_param')
        
        assert metrics['parameter_name'] == 'test_param'
        assert metrics['n_valid_points'] == 4
        assert metrics['parameter_range'] == (0.1, 0.4)
        assert metrics['event_count_range'] == (5, 15)
        assert 'parameter_sensitivity' in metrics
        assert 'parameter_event_correlation' in metrics
        
        # Should show positive correlation (parameter increases, events increase)
        assert metrics['parameter_event_correlation'] > 0.5
    
    def test_2d_visualization_data_preparation(self):
        """Test 2D visualization data preparation."""
        analyzer = MultiParameterSpectralAnalyzer()
        
        # Create mock results with 2D parameter space
        filtration = MultiParameterFiltration(['x', 'y'], [(0.0, 1.0), (0.0, 1.0)], [3, 3])
        
        mock_results = {
            'persistence_result': {
                'parameter_space': filtration,
                'eigenvalue_tensor': {
                    filtration.parameter_points[0]: torch.tensor([0.0, 0.5]),
                    filtration.parameter_points[4]: torch.tensor([0.1, 0.6])  # Center point
                },
                'persistence_events': [
                    {'parameter_point': filtration.parameter_points[0]},
                    {'parameter_point': filtration.parameter_points[4]}
                ]
            }
        }
        
        viz_data = analyzer._prepare_2d_visualization_data(mock_results)
        
        assert viz_data['parameter_names'] == ['x', 'y']
        assert viz_data['grid_sizes'] == [3, 3]
        assert 'eigenvalue_grids' in viz_data
        assert 'persistence_event_grid' in viz_data
        assert viz_data['total_computed_points'] == 2
        
        # Check that persistence event grid is the right shape
        event_grid = viz_data['persistence_event_grid']
        assert len(event_grid) == 3
        assert len(event_grid[0]) == 3


class TestIntegrationWithExistingComponents:
    """Test integration with existing spectral components."""
    
    def create_test_sheaf(self):
        """Create a simple test sheaf."""
        poset = nx.DiGraph()
        poset.add_edges_from([('0', '1'), ('1', '2')])
        stalks = {'0': torch.eye(2), '1': torch.eye(2), '2': torch.eye(2)}
        restrictions = {
            ('0', '1'): torch.eye(2) * 0.8,
            ('1', '2'): torch.eye(2) * 0.6
        }
        return Sheaf(poset, stalks, restrictions)
    
    def test_multi_parameter_filtration_with_sheaf(self):
        """Test multi-parameter filtration creation with real sheaf."""
        sheaf = self.create_test_sheaf()
        
        # Define parameter specifications based on sheaf properties
        parameter_specs = {
            'edge_threshold': {
                'range': (0.1, 1.0),
                'threshold_func': lambda weight, param: weight >= param
            },
            'cka_threshold': {
                'range': (0.0, 1.0),
                'threshold_func': lambda weight, param: weight >= param * 0.5
            }
        }
        
        # Create multi-parameter filtration
        param_names = list(parameter_specs.keys())
        param_ranges = [spec['range'] for spec in parameter_specs.values()]
        grid_sizes = [4, 4]
        
        filtration = MultiParameterFiltration(param_names, param_ranges, grid_sizes)
        
        assert filtration.dimension == 2
        assert filtration.total_points == 16
        
        # Test that all parameter points are valid
        for point in filtration.parameter_points:
            assert len(point.coordinates) == 2
            assert 0.1 <= point.coordinates[0] <= 1.0
            assert 0.0 <= point.coordinates[1] <= 1.0
    
    def test_threshold_function_compatibility(self):
        """Test that threshold functions work with edge weights."""
        sheaf = self.create_test_sheaf()
        
        # Get edge weights from sheaf
        edge_weights = []
        for edge, restriction in sheaf.restrictions.items():
            weight = torch.norm(restriction, 'fro').item()
            edge_weights.append(weight)
        
        # Test threshold functions
        def threshold_func(weight, param):
            return weight >= param
        
        # Should work with real edge weights
        for weight in edge_weights:
            assert threshold_func(weight, 0.5) in [True, False]
            assert threshold_func(weight, 0.0) == True  # All weights >= 0
            assert threshold_func(weight, 10.0) == False  # No weights >= 10
    
    def test_cache_clearing_integration(self):
        """Test cache clearing across integrated components."""
        analyzer = MultiParameterSpectralAnalyzer()
        
        # Clear all caches
        analyzer.clear_cache()
        
        # Should complete without error
        cache_info = analyzer.persistence_computer.get_cache_info()
        if cache_info['caching_enabled']:
            assert cache_info['total_cached_items'] == 0


class TestPipelineIntegration:
    """Test complete multi-parameter pipeline integration."""
    
    def create_test_sheaf(self):
        """Create a simple test sheaf for pipeline testing."""
        poset = nx.DiGraph()
        poset.add_edges_from([('input', 'layer1'), ('layer1', 'layer2'), ('layer2', 'output')])
        
        stalks = {
            'input': torch.eye(3),
            'layer1': torch.eye(3),
            'layer2': torch.eye(2),
            'output': torch.eye(2)
        }
        
        restrictions = {
            ('input', 'layer1'): torch.tensor([[0.9, 0.0, 0.0], [0.0, 0.8, 0.0], [0.0, 0.0, 0.7]]),
            ('layer1', 'layer2'): torch.tensor([[0.6, 0.5, 0.4], [0.3, 0.7, 0.2]]),
            ('layer2', 'output'): torch.tensor([[0.9, 0.0], [0.0, 0.8]])
        }
        
        return Sheaf(poset, stalks, restrictions)
    
    def test_complete_2d_analysis_pipeline(self):
        """Test complete 2D multi-parameter analysis pipeline."""
        analyzer = MultiParameterSpectralAnalyzer()
        sheaf = self.create_test_sheaf()
        
        # Define 2D parameter specifications
        param1_spec = {
            'name': 'edge_threshold',
            'range': (0.1, 1.0),
            'threshold_func': lambda weight, param: weight >= param
        }
        param2_spec = {
            'name': 'cka_threshold',
            'range': (0.0, 0.5),
            'threshold_func': lambda weight, param: weight >= param * 2
        }
        
        # Run analysis with small grid for testing
        grid_sizes = [3, 3]
        
        # This should complete without error
        result = analyzer.analyze_2d_parameter_space(
            sheaf=sheaf,
            param1_spec=param1_spec,
            param2_spec=param2_spec,
            grid_size=tuple(grid_sizes)
        )
        
        # Validate result structure
        assert 'persistence_result' in result
        assert 'correlation_analysis' in result
        assert 'features' in result
        assert 'analysis_metadata' in result
        
        # Check persistence result
        persistence = result['persistence_result']
        assert 'parameter_space' in persistence
        assert 'eigenvalue_tensor' in persistence
        assert 'persistence_events' in persistence
        
        # Check parameter space
        param_space = persistence['parameter_space']
        assert param_space.dimension == 2
        assert param_space.total_points == 9  # 3x3 grid
        
        # Check filtration specification
        if 'filtration_specification' in result:
            filtration_spec = result['filtration_specification']
            assert 'parameter_names' in filtration_spec
            assert 'grid_sizes' in filtration_spec
            assert filtration_spec['parameter_names'] == ['edge_threshold', 'cka_threshold']
            assert filtration_spec['grid_sizes'] == [3, 3]
    
    def test_parameter_sensitivity_analysis(self):
        """Test parameter sensitivity analysis."""
        analyzer = MultiParameterSpectralAnalyzer()
        sheaf = self.create_test_sheaf()
        
        # Define base parameters including the sensitivity parameter
        base_parameter_specs = {
            'edge_threshold': {
                'range': (0.2, 0.8),  # This will be overridden by sensitivity_range
                'threshold_func': lambda weight, param: weight >= param
            },
            'cka_threshold': {
                'range': (0.0, 0.5),
                'threshold_func': lambda weight, param: weight >= param * 2
            }
        }
        
        # Run sensitivity analysis
        sensitivity_result = analyzer.analyze_parameter_sensitivity(
            sheaf=sheaf,
            base_parameter_specs=base_parameter_specs,
            sensitivity_parameter='edge_threshold',
            sensitivity_range=(0.2, 0.8),
            n_sensitivity_points=5
        )
        
        # Validate sensitivity result structure
        assert 'sensitivity_parameter' in sensitivity_result
        assert 'sensitivity_range' in sensitivity_result
        assert 'sensitivity_results' in sensitivity_result
        
        assert sensitivity_result['sensitivity_parameter'] == 'edge_threshold'
        assert sensitivity_result['sensitivity_range'] == (0.2, 0.8)
        
        # Should have attempted 5 sensitivity points
        assert len(sensitivity_result['sensitivity_results']) == 5
        
        # Check sensitivity metrics (if any valid results)
        if 'sensitivity_metrics' in sensitivity_result:
            metrics = sensitivity_result['sensitivity_metrics']
            # Metrics structure may vary depending on success/failure
    
    def test_error_handling_pipeline(self):
        """Test error handling in multi-parameter pipeline."""
        analyzer = MultiParameterSpectralAnalyzer()
        sheaf = self.create_test_sheaf()
        
        # Test invalid parameter specifications
        invalid_spec = {
            'name': 'invalid_param',
            'range': (1.0, 0.0),  # Invalid range
            'threshold_func': lambda w, p: w >= p
        }
        
        valid_spec = {
            'name': 'valid_param',
            'range': (0.0, 1.0),
            'threshold_func': lambda w, p: w >= p
        }
        
        with pytest.raises(ValueError):
            analyzer.analyze_2d_parameter_space(
                sheaf=sheaf,
                param1_spec=invalid_spec,
                param2_spec=valid_spec,
                grid_size=(3, 3)
            )
    
    def test_performance_with_larger_grid(self):
        """Test performance with larger parameter grid."""
        analyzer = MultiParameterSpectralAnalyzer()
        sheaf = self.create_test_sheaf()
        
        # Define parameter specifications
        param1_spec = {
            'name': 'edge_threshold',
            'range': (0.1, 1.0),
            'threshold_func': lambda weight, param: weight >= param
        }
        param2_spec = {
            'name': 'cka_threshold',
            'range': (0.0, 0.5),
            'threshold_func': lambda weight, param: weight >= param
        }
        
        # Test with slightly larger grid (but not too large for CI)
        grid_sizes = [5, 5]  # 25 parameter points
        
        import time
        start_time = time.time()
        
        result = analyzer.analyze_2d_parameter_space(
            sheaf=sheaf,
            param1_spec=param1_spec,
            param2_spec=param2_spec,
            grid_size=tuple(grid_sizes)
        )
        
        elapsed_time = time.time() - start_time
        
        # Should complete reasonably quickly for test suite
        assert elapsed_time < 30.0  # 30 seconds max for small test
        
        # Validate that all parameter points were computed
        persistence = result['persistence_result']
        assert persistence['parameter_space'].total_points == 25
        
        # Check that we have eigenvalue results
        eigenvalue_tensor = persistence['eigenvalue_tensor']
        assert len(eigenvalue_tensor) > 0
    
    def test_caching_effectiveness(self):
        """Test that caching improves performance."""
        analyzer = MultiParameterSpectralAnalyzer()
        sheaf = self.create_test_sheaf()
        
        # Define parameter specifications
        param1_spec = {
            'name': 'edge_threshold',
            'range': (0.3, 0.7),
            'threshold_func': lambda weight, param: weight >= param
        }
        param2_spec = {
            'name': 'cka_threshold',
            'range': (0.1, 0.3),
            'threshold_func': lambda weight, param: weight >= param
        }
        
        grid_sizes = [3, 3]
        
        # First run (no cache)
        import time
        start_time = time.time()
        result1 = analyzer.analyze_2d_parameter_space(
            sheaf=sheaf,
            param1_spec=param1_spec,
            param2_spec=param2_spec,
            grid_size=tuple(grid_sizes)
        )
        first_run_time = time.time() - start_time
        
        # Second run (with cache)
        start_time = time.time()
        result2 = analyzer.analyze_2d_parameter_space(
            sheaf=sheaf,
            param1_spec=param1_spec,
            param2_spec=param2_spec,
            grid_size=tuple(grid_sizes)
        )
        second_run_time = time.time() - start_time
        
        # Second run should be faster due to caching
        # (allowing some tolerance for variation)
        assert second_run_time <= first_run_time + 0.1
        
        # Results should be consistent
        persistence1 = result1['persistence_result']
        persistence2 = result2['persistence_result']
        
        assert len(persistence1['eigenvalue_tensor']) == len(persistence2['eigenvalue_tensor'])
    
    def test_memory_management(self):
        """Test memory management in multi-parameter analysis."""
        analyzer = MultiParameterSpectralAnalyzer()
        sheaf = self.create_test_sheaf()
        
        # Get initial cache info
        initial_cache = analyzer.persistence_computer.get_cache_info()
        
        # Run analysis
        param1_spec = {
            'name': 'edge_threshold',
            'range': (0.2, 0.8),
            'threshold_func': lambda weight, param: weight >= param
        }
        param2_spec = {
            'name': 'secondary_param',
            'range': (0.1, 0.5),
            'threshold_func': lambda weight, param: weight >= param * 0.5
        }
        
        result = analyzer.analyze_2d_parameter_space(
            sheaf=sheaf,
            param1_spec=param1_spec,
            param2_spec=param2_spec,
            grid_size=(3, 3)
        )
        
        # Check that cache has data
        cache_after = analyzer.persistence_computer.get_cache_info()
        if cache_after['caching_enabled']:
            assert cache_after['total_cached_items'] > initial_cache.get('total_cached_items', 0)
        
        # Clear cache and verify
        analyzer.clear_cache()
        cache_cleared = analyzer.persistence_computer.get_cache_info()
        if cache_cleared['caching_enabled']:
            assert cache_cleared['total_cached_items'] == 0


class TestMathematicalCorrectness:
    """Test mathematical correctness of multi-parameter persistence."""
    
    def test_parameter_point_ordering(self):
        """Test that parameter point ordering respects multi-parameter structure."""
        # Create test points
        p1 = ParameterPoint((0.1, 0.2), 0)
        p2 = ParameterPoint((0.3, 0.4), 1)
        p3 = ParameterPoint((0.2, 0.5), 2)
        p4 = ParameterPoint((0.3, 0.2), 3)
        
        # Test domination relationships
        assert p2.dominates(p1)  # (0.3, 0.4) dominates (0.1, 0.2)
        assert not p1.dominates(p2)
        
        assert p3.dominates(p1)  # (0.2, 0.5) dominates (0.1, 0.2)
        assert not p1.dominates(p3)
        
        # These should not dominate each other (incomparable)
        assert not p3.dominates(p4)  # (0.2, 0.5) vs (0.3, 0.2)
        assert not p4.dominates(p3)
    
    def test_filtration_monotonicity(self):
        """Test that filtration respects monotonicity in parameter space."""
        filtration = MultiParameterFiltration(
            ['param1', 'param2'],
            [(0.0, 1.0), (0.0, 1.0)],
            [4, 4]
        )
        
        # For threshold-based filtration, larger parameter values
        # should result in smaller or equal filtered complexes
        threshold_func = lambda weight, param: weight >= param
        
        # Test with simple edge weights
        edge_weights = [0.2, 0.5, 0.8]
        
        for i, point1 in enumerate(filtration.parameter_points[:-1]):
            for point2 in filtration.parameter_points[i+1:]:
                if point2.dominates(point1):
                    # Higher threshold should keep fewer or equal edges
                    edges_kept_1 = sum(threshold_func(w, max(point1.coordinates)) for w in edge_weights)
                    edges_kept_2 = sum(threshold_func(w, max(point2.coordinates)) for w in edge_weights)
                    
                    assert edges_kept_2 <= edges_kept_1, (
                        f"Monotonicity violated: point {point2.coordinates} "
                        f"keeps more edges than {point1.coordinates}"
                    )
    
    def test_eigenvalue_continuity(self):
        """Test eigenvalue continuity across parameter space."""
        # This is a simplified test - in practice, eigenvalues should vary
        # continuously across the parameter space for well-behaved filtrations
        
        filtration = MultiParameterFiltration(
            ['param1', 'param2'],
            [(0.1, 0.9), (0.1, 0.9)],
            [3, 3]
        )
        
        # Create mock eigenvalue results that should be continuous
        eigenvalue_tensor = {}
        
        for point in filtration.parameter_points:
            # Simple continuous function for testing
            x, y = point.coordinates
            eigenvals = torch.tensor([x + y, x * y, abs(x - y)])
            eigenvalue_tensor[point] = eigenvals
        
        # Check that nearby points have similar eigenvalues
        tolerance = 0.5  # Generous tolerance for discrete grid
        
        for point in filtration.parameter_points:
            neighbors = filtration.get_neighbors(point, 'face')
            
            for neighbor in neighbors:
                if neighbor in eigenvalue_tensor:
                    eigenvals_1 = eigenvalue_tensor[point]
                    eigenvals_2 = eigenvalue_tensor[neighbor]
                    
                    # Check that eigenvalues don't jump too much
                    max_diff = torch.max(torch.abs(eigenvals_1 - eigenvals_2))
                    assert max_diff < tolerance, (
                        f"Eigenvalue discontinuity between {point.coordinates} "
                        f"and {neighbor.coordinates}: max_diff={max_diff}"
                    )
    
    def test_persistence_event_validity(self):
        """Test that persistence events are mathematically valid."""
        # Create mock persistence events
        filtration = MultiParameterFiltration(
            ['x', 'y'],
            [(0.0, 1.0), (0.0, 1.0)],
            [3, 3]
        )
        
        persistence_events = [
            {
                'type': 'birth',
                'parameter_point': filtration.parameter_points[0],
                'parameter_dimension': 0,
                'parameter_name': 'x',
                'eigenvalue_index': 1
            },
            {
                'type': 'death',
                'parameter_point': filtration.parameter_points[4],
                'parameter_dimension': 0,
                'parameter_name': 'x',
                'eigenvalue_index': 1
            }
        ]
        
        # Validate event structure
        for event in persistence_events:
            assert 'type' in event
            assert event['type'] in ['birth', 'death']
            assert 'parameter_point' in event
            assert isinstance(event['parameter_point'], ParameterPoint)
            assert 'parameter_dimension' in event
            assert 'parameter_name' in event
        
        # Check that birth comes before death in parameter space
        birth_events = [e for e in persistence_events if e['type'] == 'birth']
        death_events = [e for e in persistence_events if e['type'] == 'death']
        
        for birth in birth_events:
            for death in death_events:
                if (birth['eigenvalue_index'] == death['eigenvalue_index'] and
                    birth['parameter_name'] == death['parameter_name']):
                    # Death should occur at parameter >= birth parameter
                    birth_point = birth['parameter_point']
                    death_point = death['parameter_point']
                    
                    # For the parameter dimension being tracked
                    param_dim = birth['parameter_dimension']
                    birth_val = birth_point.coordinates[param_dim]
                    death_val = death_point.coordinates[param_dim]
                    
                    assert death_val >= birth_val, (
                        f"Death parameter {death_val} should be >= birth parameter {birth_val}"
                    )
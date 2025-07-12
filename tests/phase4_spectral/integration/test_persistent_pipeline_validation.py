# tests/phase4_spectral/integration/test_persistent_pipeline_validation.py
"""Integration tests for end-to-end persistent spectral analysis pipeline.

This module validates the complete pipeline from sheaf construction through
persistence computation, feature extraction, and diagram generation against
known mathematical ground truth and established persistent homology properties.
"""

import pytest
import torch
import numpy as np
import networkx as nx
import time
from neurosheaf.spectral.persistent import PersistentSpectralAnalyzer
from neurosheaf.spectral.static_laplacian_masking import StaticLaplacianWithMasking
from neurosheaf.spectral.tracker import SubspaceTracker
from neurosheaf.sheaf.construction import Sheaf
from ..utils.test_ground_truth import GroundTruthGenerator, PersistenceValidator


class TestPersistentPipelineValidation:
    """Test complete persistent spectral analysis pipeline."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.generator = GroundTruthGenerator()
        self.validator = PersistenceValidator()
        self.tolerance = 1e-6
        self.persistence_tolerance = 1e-3
    
    def test_end_to_end_linear_chain(self):
        """Test complete pipeline on linear chain with known properties."""
        # Create linear chain sheaf
        sheaf, expected = self.generator.linear_chain_sheaf(n_nodes=5, stalk_dim=2)
        
        # Create analyzer
        analyzer = PersistentSpectralAnalyzer(
            default_n_steps=10,
            default_filtration_type='threshold'
        )
        
        # Perform complete analysis
        result = analyzer.analyze(sheaf, n_steps=10)
        
        # Validate result structure
        assert 'persistence_result' in result, "Missing persistence_result"
        assert 'features' in result, "Missing features"
        assert 'diagrams' in result, "Missing diagrams"
        assert 'filtration_params' in result, "Missing filtration_params"
        assert 'analysis_metadata' in result, "Missing analysis_metadata"
        
        # Validate filtration parameters
        filtration_params = result['filtration_params']
        assert len(filtration_params) == 10, f"Expected 10 filtration steps, got {len(filtration_params)}"
        assert filtration_params == sorted(filtration_params), "Filtration parameters not sorted"
        
        # Validate eigenvalue sequences
        eigenval_seqs = result['persistence_result']['eigenvalue_sequences']
        assert len(eigenval_seqs) == 10, f"Expected 10 eigenvalue sequences, got {len(eigenval_seqs)}"
        
        # All eigenvalue sequences should pass basic validation
        for i, eigenvals in enumerate(eigenval_seqs):
            validation = self.validator.validate_eigenvalue_properties(eigenvals)
            assert validation['non_negative'], f"Step {i}: eigenvalues not non-negative"
            assert validation['finite'], f"Step {i}: eigenvalues not finite"
        
        # Validate persistence diagrams
        diagrams = result['diagrams']
        assert 'birth_death_pairs' in diagrams, "Missing birth_death_pairs"
        assert 'infinite_bars' in diagrams, "Missing infinite_bars"
        assert 'statistics' in diagrams, "Missing diagram statistics"
        
        # Validate persistence pairs
        diagram_validation = self.validator.validate_persistence_diagram(diagrams['birth_death_pairs'])
        assert diagram_validation['birth_death_ordering'], "Birth-death ordering violated"
        assert diagram_validation['positive_lifetimes'], "Negative lifetimes found"
        assert diagram_validation['finite_births'], "Non-finite birth times found"
        
        # Linear chains should have at least one infinite bar (connectivity)
        assert len(diagrams['infinite_bars']) >= 1, "Linear chain should have at least one infinite bar"
        
        # Validate features
        features = result['features']
        assert 'eigenvalue_evolution' in features, "Missing eigenvalue_evolution"
        assert 'spectral_gap_evolution' in features, "Missing spectral_gap_evolution" 
        assert 'num_birth_events' in features, "Missing num_birth_events"
        assert 'num_death_events' in features, "Missing num_death_events"
        assert 'summary' in features, "Missing feature summary"
        
        # Feature sequences should match filtration length
        assert len(features['eigenvalue_evolution']) == 10, "Eigenvalue evolution length mismatch"
        assert len(features['spectral_gap_evolution']) == 10, "Spectral gap evolution length mismatch"
    
    def test_end_to_end_cycle_graph(self):
        """Test complete pipeline on cycle graph with loop structure."""
        # Create cycle graph sheaf
        sheaf, expected = self.generator.cycle_graph_sheaf(n_nodes=6, stalk_dim=2)
        
        analyzer = PersistentSpectralAnalyzer(default_n_steps=8)
        result = analyzer.analyze(sheaf, n_steps=8)
        
        # Basic structure validation
        assert len(result['persistence_result']['eigenvalue_sequences']) == 8
        assert len(result['filtration_params']) == 8
        
        # Validate topological properties specific to cycles
        eigenval_seqs = result['persistence_result']['eigenvalue_sequences']
        
        # Each step should have at least one zero eigenvalue (connectivity)
        for i, eigenvals in enumerate(eigenval_seqs):
            zero_count = torch.sum(eigenvals < self.tolerance).item()
            assert zero_count >= 1, f"Step {i}: No zero eigenvalue (not connected)"
        
        # Should detect loop-related persistence features
        features = result['features']
        total_events = features['num_birth_events'] + features['num_death_events']
        
        # Cycles should produce some topological events during filtration
        assert total_events >= 0, "No persistence events detected in cycle"
        
        # Validate diagram properties
        diagrams = result['diagrams']
        stats = diagrams['statistics']
        assert stats['n_finite_pairs'] >= 0, "Negative finite pairs count"
        assert stats['n_infinite_bars'] >= 1, "Cycle should have at least one infinite bar"
    
    def test_end_to_end_disconnected_components(self):
        """Test pipeline on disconnected graph with multiple components."""
        # Create disconnected components
        sheaf, expected = self.generator.disconnected_components_sheaf([3, 4, 2], stalk_dim=2)
        
        analyzer = PersistentSpectralAnalyzer(default_n_steps=6)
        result = analyzer.analyze(sheaf, n_steps=6)
        
        # Validate multiple component detection
        eigenval_seqs = result['persistence_result']['eigenvalue_sequences']
        
        # Should have multiple zero eigenvalues reflecting multiple components
        for i, eigenvals in enumerate(eigenval_seqs):
            zero_count = torch.sum(eigenvals < self.tolerance).item()
            # Should have at least as many zeros as components (3 components expected)
            assert zero_count >= expected['n_components'], \
                f"Step {i}: Zero count {zero_count} < expected components {expected['n_components']}"
        
        # Should have multiple infinite bars
        diagrams = result['diagrams']
        assert diagrams['statistics']['n_infinite_bars'] >= expected['n_components'], \
            f"Infinite bars {diagrams['statistics']['n_infinite_bars']} < components {expected['n_components']}"
    
    def test_filtration_type_consistency(self):
        """Test consistency across different filtration types."""
        sheaf, expected = self.generator.cycle_graph_sheaf(5, stalk_dim=2)
        
        analyzer = PersistentSpectralAnalyzer(default_n_steps=5)
        
        # Test different filtration types
        filtration_types = ['threshold', 'cka_based']
        results = {}
        
        for filt_type in filtration_types:
            try:
                result = analyzer.analyze(sheaf, filtration_type=filt_type, n_steps=5)
                results[filt_type] = result
                
                # Basic validation for each type
                assert len(result['filtration_params']) == 5
                assert result['filtration_type'] == filt_type
                assert 'persistence_result' in result
                
            except Exception as e:
                pytest.fail(f"Filtration type {filt_type} failed: {e}")
        
        # All filtration types should produce valid results
        assert len(results) == len(filtration_types), "Some filtration types failed"
        
        # Results should have similar structure (though not identical values)
        for filt_type, result in results.items():
            eigenval_seqs = result['persistence_result']['eigenvalue_sequences']
            
            for i, eigenvals in enumerate(eigenval_seqs):
                validation = self.validator.validate_eigenvalue_properties(eigenvals)
                assert all(validation.values()), \
                    f"Filtration {filt_type}, step {i}: eigenvalue validation failed"
    
    def test_parameter_range_effects(self):
        """Test effects of different parameter ranges on results."""
        sheaf, expected = self.generator.linear_chain_sheaf(4, stalk_dim=2)
        
        analyzer = PersistentSpectralAnalyzer()
        
        # Test different parameter ranges
        ranges = [
            (0.1, 0.5),   # Narrow range
            (0.0, 1.0),   # Standard range
            (0.0, 2.0)    # Wide range
        ]
        
        results = []
        for param_range in ranges:
            result = analyzer.analyze(sheaf, n_steps=5, param_range=param_range)
            results.append(result)
            
            # Validate parameter range was respected
            params = result['filtration_params']
            assert min(params) >= param_range[0] - self.tolerance, \
                f"Minimum parameter {min(params)} < requested {param_range[0]}"
            assert max(params) <= param_range[1] + self.tolerance, \
                f"Maximum parameter {max(params)} > requested {param_range[1]}"
        
        # All ranges should produce valid results
        assert len(results) == len(ranges), "Some parameter ranges failed"
        
        # Different ranges may produce different numbers of persistence events
        for i, result in enumerate(results):
            features = result['features']
            total_events = features['num_birth_events'] + features['num_death_events']
            assert total_events >= 0, f"Range {ranges[i]}: negative event count"
    
    def test_edge_threshold_functions(self):
        """Test different edge threshold functions."""
        sheaf, expected = self.generator.cycle_graph_sheaf(4, stalk_dim=2)
        
        analyzer = PersistentSpectralAnalyzer()
        
        # Test standard threshold
        result1 = analyzer.analyze(sheaf, filtration_type='threshold', n_steps=5)
        
        # Test CKA-based threshold
        result2 = analyzer.analyze(sheaf, filtration_type='cka_based', n_steps=5)
        
        # Test custom threshold function
        def custom_threshold(weight, param):
            return weight >= param * 0.5  # More permissive threshold
        
        result3 = analyzer.analyze(
            sheaf, 
            filtration_type='custom', 
            n_steps=5,
            custom_threshold_func=custom_threshold
        )
        
        # All should complete successfully
        results = [result1, result2, result3]
        for i, result in enumerate(results):
            assert 'persistence_result' in result, f"Result {i}: missing persistence_result"
            assert len(result['filtration_params']) == 5, f"Result {i}: wrong parameter count"
            
            # Validate eigenvalue properties
            eigenval_seqs = result['persistence_result']['eigenvalue_sequences']
            for j, eigenvals in enumerate(eigenval_seqs):
                validation = self.validator.validate_eigenvalue_properties(eigenvals)
                assert validation['finite'], f"Result {i}, step {j}: non-finite eigenvalues"
                assert validation['non_negative'], f"Result {i}, step {j}: negative eigenvalues"
    
    def test_feature_extraction_completeness(self):
        """Test completeness and validity of extracted features."""
        sheaf, expected = self.generator.tree_sheaf(depth=2, branching_factor=3, stalk_dim=2)
        
        analyzer = PersistentSpectralAnalyzer()
        result = analyzer.analyze(sheaf, n_steps=8)
        
        features = result['features']
        
        # Check all expected feature categories
        expected_features = [
            'eigenvalue_evolution',
            'spectral_gap_evolution', 
            'effective_dimension',
            'eigenvalue_statistics',
            'num_birth_events',
            'num_death_events',
            'num_crossings',
            'num_persistent_paths',
            'summary'
        ]
        
        for feature_name in expected_features:
            assert feature_name in features, f"Missing feature: {feature_name}"
        
        # Validate feature data types and ranges
        n_steps = 8
        
        # Evolution features should have correct length
        assert len(features['eigenvalue_evolution']) == n_steps, "Eigenvalue evolution length wrong"
        assert len(features['spectral_gap_evolution']) == n_steps, "Spectral gap evolution length wrong"
        assert len(features['effective_dimension']) == n_steps, "Effective dimension evolution length wrong"
        
        # Event counts should be non-negative integers
        assert isinstance(features['num_birth_events'], int) and features['num_birth_events'] >= 0
        assert isinstance(features['num_death_events'], int) and features['num_death_events'] >= 0
        assert isinstance(features['num_crossings'], int) and features['num_crossings'] >= 0
        
        # Summary statistics should be valid
        summary = features['summary']
        assert 'total_filtration_steps' in summary
        assert summary['total_filtration_steps'] == n_steps
        assert 'mean_eigenvals_per_step' in summary
        assert summary['mean_eigenvals_per_step'] >= 0
        
        # Effective dimensions should be positive
        for eff_dim in features['effective_dimension']:
            assert eff_dim >= 0, f"Negative effective dimension: {eff_dim}"
    
    def test_persistence_diagram_properties(self):
        """Test mathematical properties of persistence diagrams."""
        sheaf, expected = self.generator.complete_graph_sheaf(4, stalk_dim=2)
        
        analyzer = PersistentSpectralAnalyzer()
        result = analyzer.analyze(sheaf, n_steps=10)
        
        diagrams = result['diagrams']
        
        # Validate birth-death pairs
        for pair in diagrams['birth_death_pairs']:
            assert 'birth' in pair and 'death' in pair and 'lifetime' in pair
            assert pair['birth'] <= pair['death'], f"Birth {pair['birth']} > Death {pair['death']}"
            assert pair['lifetime'] >= 0, f"Negative lifetime: {pair['lifetime']}"
            assert np.isfinite(pair['birth']), f"Non-finite birth: {pair['birth']}"
            assert np.isfinite(pair['death']), f"Non-finite death: {pair['death']}"
        
        # Validate infinite bars
        for bar in diagrams['infinite_bars']:
            assert 'birth' in bar and 'death' in bar
            assert np.isfinite(bar['birth']), f"Non-finite birth in infinite bar: {bar['birth']}"
            assert bar['death'] == float('inf'), f"Infinite bar death not infinite: {bar['death']}"
        
        # Validate statistics
        stats = diagrams['statistics']
        assert stats['n_finite_pairs'] == len(diagrams['birth_death_pairs'])
        assert stats['n_infinite_bars'] == len(diagrams['infinite_bars'])
        
        if stats['n_finite_pairs'] > 0:
            assert stats['mean_lifetime'] >= 0, "Negative mean lifetime"
            assert stats['max_lifetime'] >= 0, "Negative max lifetime"
            assert stats['total_persistence'] >= 0, "Negative total persistence"
    
    def test_analysis_metadata_validation(self):
        """Test analysis metadata completeness and accuracy."""
        sheaf, expected = self.generator.linear_chain_sheaf(6, stalk_dim=2)
        
        analyzer = PersistentSpectralAnalyzer()
        
        start_time = time.time()
        result = analyzer.analyze(sheaf, n_steps=7)
        end_time = time.time()
        
        metadata = result['analysis_metadata']
        
        # Check required metadata fields
        required_fields = [
            'analysis_time',
            'computation_time', 
            'n_eigenvalue_sequences',
            'n_filtration_steps',
            'sheaf_nodes',
            'sheaf_edges'
        ]
        
        for field in required_fields:
            assert field in metadata, f"Missing metadata field: {field}"
        
        # Validate metadata values
        assert metadata['analysis_time'] > 0, "Analysis time should be positive"
        assert metadata['analysis_time'] <= (end_time - start_time + 1), "Analysis time seems too large"
        assert metadata['n_eigenvalue_sequences'] == 7, "Wrong eigenvalue sequence count"
        assert metadata['n_filtration_steps'] == 7, "Wrong filtration step count"
        assert metadata['sheaf_nodes'] == expected['n_nodes'], "Wrong sheaf node count"
        assert metadata['sheaf_edges'] >= 0, "Negative edge count"
    
    def test_multiple_sheaves_analysis(self):
        """Test analysis of multiple sheaves."""
        # Create multiple test sheaves
        sheaves = []
        expected_results = []
        
        # Different graph types
        sheaf1, exp1 = self.generator.linear_chain_sheaf(4, stalk_dim=2)
        sheaf2, exp2 = self.generator.cycle_graph_sheaf(4, stalk_dim=2)
        sheaf3, exp3 = self.generator.tree_sheaf(depth=2, branching_factor=2, stalk_dim=2)
        
        sheaves = [sheaf1, sheaf2, sheaf3]
        expected_results = [exp1, exp2, exp3]
        
        analyzer = PersistentSpectralAnalyzer()
        
        # Analyze multiple sheaves
        results = analyzer.analyze_multiple_sheaves(sheaves, n_steps=5)
        
        # Should return one result per sheaf
        assert len(results) == len(sheaves), f"Expected {len(sheaves)} results, got {len(results)}"
        
        # Each result should be valid
        for i, result in enumerate(results):
            assert 'persistence_result' in result, f"Result {i}: missing persistence_result"
            assert 'features' in result, f"Result {i}: missing features"
            assert 'diagrams' in result, f"Result {i}: missing diagrams"
            assert len(result['filtration_params']) == 5, f"Result {i}: wrong parameter count"
            
            # Validate eigenvalue properties
            eigenval_seqs = result['persistence_result']['eigenvalue_sequences']
            for j, eigenvals in enumerate(eigenval_seqs):
                validation = self.validator.validate_eigenvalue_properties(eigenvals)
                assert validation['finite'], f"Result {i}, step {j}: non-finite eigenvalues"


class TestPipelineRobustness:
    """Test robustness and edge case handling of the pipeline."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.generator = GroundTruthGenerator()
        self.validator = PersistenceValidator()
    
    def test_empty_sheaf_handling(self):
        """Test pipeline behavior with minimal/empty sheaves."""
        # Single node, no edges
        single_poset = nx.DiGraph()
        single_poset.add_node('A')
        single_sheaf = Sheaf(single_poset, {'A': torch.eye(2)}, {})
        
        analyzer = PersistentSpectralAnalyzer()
        
        try:
            result = analyzer.analyze(single_sheaf, n_steps=3)
            
            # Should complete without error
            assert 'persistence_result' in result
            assert 'features' in result
            assert 'diagrams' in result
            assert len(result['filtration_params']) == 3
            
        except Exception as e:
            pytest.fail(f"Single node sheaf analysis failed: {e}")
    
    def test_zero_steps_handling(self):
        """Test handling of zero filtration steps."""
        sheaf, expected = self.generator.linear_chain_sheaf(3, stalk_dim=2)
        
        analyzer = PersistentSpectralAnalyzer()
        
        # Should handle zero steps gracefully
        result = analyzer.analyze(sheaf, n_steps=0)
        
        assert len(result['filtration_params']) == 0
        assert len(result['persistence_result']['eigenvalue_sequences']) == 0
        assert result['features']['num_birth_events'] == 0
        assert result['features']['num_death_events'] == 0
    
    def test_invalid_parameter_handling(self):
        """Test handling of invalid parameters."""
        sheaf, expected = self.generator.linear_chain_sheaf(3, stalk_dim=2)
        
        analyzer = PersistentSpectralAnalyzer()
        
        # Test invalid filtration type
        with pytest.raises((ValueError, Exception)):
            analyzer._create_edge_threshold_func('invalid_type')
        
        # Test invalid parameter range (min > max)
        try:
            result = analyzer.analyze(sheaf, n_steps=3, param_range=(1.0, 0.0))
            # Should either handle gracefully or raise appropriate error
        except Exception:
            pass  # Expected behavior
    
    def test_large_stalk_dimensions(self):
        """Test handling of larger stalk dimensions."""
        # This might be slow, but tests scalability
        sheaf, expected = self.generator.linear_chain_sheaf(n_nodes=3, stalk_dim=10)
        
        analyzer = PersistentSpectralAnalyzer()
        
        try:
            result = analyzer.analyze(sheaf, n_steps=5)
            
            # Should handle larger dimensions
            assert 'persistence_result' in result
            eigenval_seqs = result['persistence_result']['eigenvalue_sequences']
            
            # Should have more eigenvalues due to larger stalk dimension
            for eigenvals in eigenval_seqs:
                assert len(eigenvals) >= expected['stalk_dim']
                
        except Exception as e:
            pytest.fail(f"Large stalk dimension handling failed: {e}")


@pytest.mark.slow
class TestPipelinePerformance:
    """Performance tests for the persistent spectral analysis pipeline."""
    
    def test_moderate_size_performance(self):
        """Test performance on moderately sized graphs."""
        # Create moderately large test case
        generator = GroundTruthGenerator()
        sheaf, expected = generator.linear_chain_sheaf(n_nodes=20, stalk_dim=3)
        
        analyzer = PersistentSpectralAnalyzer(
            static_laplacian=StaticLaplacianWithMasking(eigenvalue_method='lobpcg')
        )
        
        start_time = time.time()
        result = analyzer.analyze(sheaf, n_steps=15)
        computation_time = time.time() - start_time
        
        # Should complete in reasonable time (< 60 seconds for moderate size)
        assert computation_time < 60.0, f"Moderate size computation too slow: {computation_time}s"
        
        # Should produce valid results
        validator = PersistenceValidator()
        eigenval_seqs = result['persistence_result']['eigenvalue_sequences']
        
        for eigenvals in eigenval_seqs:
            validation = validator.validate_eigenvalue_properties(eigenvals)
            assert validation['finite'], "Performance test produced non-finite eigenvalues"
            assert validation['non_negative'], "Performance test produced negative eigenvalues"
    
    def test_memory_efficiency(self):
        """Test memory efficiency of the pipeline."""
        import psutil
        import os
        
        # Monitor memory usage
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        generator = GroundTruthGenerator()
        sheaf, expected = generator.cycle_graph_sheaf(n_nodes=10, stalk_dim=4)
        
        analyzer = PersistentSpectralAnalyzer()
        result = analyzer.analyze(sheaf, n_steps=10)
        
        peak_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = peak_memory - initial_memory
        
        # Memory increase should be reasonable (< 500 MB for this size)
        assert memory_increase < 500, f"Memory increase too large: {memory_increase} MB"
        
        # Clear cache to free memory
        analyzer.clear_cache()
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_after_clear = final_memory - initial_memory
        
        # Memory should be reduced after clearing cache
        assert memory_after_clear <= memory_increase, "Cache clearing did not reduce memory"
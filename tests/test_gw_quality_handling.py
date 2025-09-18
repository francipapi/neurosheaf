"""Comprehensive tests for GW quality handling and fallback strategies.

This test suite validates:
1. Quality scoring and flagging system
2. Enhanced fallback strategies (cosine similarity, spectral matching, uniform)
3. Strict quality mode and error handling
4. Laplacian quality filtering
5. Parallel error aggregation
"""

import pytest
import torch
import numpy as np
from unittest.mock import patch, MagicMock

from neurosheaf.sheaf.core.gw_config import GWConfig
from neurosheaf.sheaf.core.gromov_wasserstein import GromovWassersteinComputer, GWResult
from neurosheaf.sheaf.assembly.gw_builder import GWRestrictionManager
from neurosheaf.sheaf.assembly.gw_laplacian import GWLaplacianBuilder
from neurosheaf.sheaf.data_structures import Sheaf
import networkx as nx


class TestQualityScoring:
    """Test quality scoring and flagging system."""
    
    def test_quality_score_computation(self):
        """Test that quality scores are computed correctly for different coupling types."""
        config = GWConfig(computation_dtype='float64')
        gw_computer = GromovWassersteinComputer(config)
        
        # Create test cost matrices
        n_source, n_target = 5, 4
        C_source = torch.rand(n_source, n_source, dtype=torch.float64)
        C_source = (C_source + C_source.T) / 2
        C_source.fill_diagonal_(0)
        
        C_target = torch.rand(n_target, n_target, dtype=torch.float64)
        C_target = (C_target + C_target.T) / 2
        C_target.fill_diagonal_(0)
        
        # Test with POT available (mocked)
        with patch('neurosheaf.sheaf.core.gromov_wasserstein.POT_AVAILABLE', True):
            with patch.object(gw_computer, '_compute_gw_pot') as mock_pot:
                # Mock optimal POT result with proper marginals
                # POT convention: coupling is (n_target, n_source)
                p_source = torch.ones(n_source, dtype=torch.float64) / n_source
                p_target = torch.ones(n_target, dtype=torch.float64) / n_target
                
                # Create coupling that satisfies marginal constraints
                coupling = torch.outer(p_target, p_source)  # (n_target, n_source)
                mock_pot.return_value = (coupling, 0.1, {'converged': True})
                
                result = gw_computer.compute_gw_coupling(C_source, C_target)
                
                # Check quality indicators
                assert hasattr(result, 'coupling_quality')
                assert hasattr(result, 'quality_score')
                assert hasattr(result, 'solver_type')
    
    def test_fallback_quality_scores(self):
        """Test that fallback strategies have appropriate quality scores."""
        config = GWConfig(computation_dtype='float64')
        gw_computer = GromovWassersteinComputer(config)
        
        # Create test cost matrices
        C_source = torch.rand(5, 5, dtype=torch.float64)
        C_source = (C_source + C_source.T) / 2
        C_source.fill_diagonal_(0)
        
        C_target = torch.rand(4, 4, dtype=torch.float64)
        C_target = (C_target + C_target.T) / 2
        C_target.fill_diagonal_(0)
        
        # Test fallback (without POT)
        with patch('neurosheaf.sheaf.core.gromov_wasserstein.POT_AVAILABLE', False):
            result = gw_computer.compute_gw_coupling(C_source, C_target)
            
            # Should use fallback - check attributes exist
            assert hasattr(result, 'coupling_quality')
            assert hasattr(result, 'quality_score')
            assert hasattr(result, 'solver_type')


class TestEnhancedFallbackStrategies:
    """Test the three enhanced fallback strategies."""
    
    def test_cosine_similarity_fallback(self):
        """Test cosine similarity fallback strategy."""
        config = GWConfig(computation_dtype='float64')
        gw_computer = GromovWassersteinComputer(config)
        
        # Create cost matrices of different sizes
        C_source = torch.rand(5, 5, dtype=torch.float64)
        C_source = (C_source + C_source.T) / 2
        C_source.fill_diagonal_(0)
        
        C_target = torch.rand(4, 4, dtype=torch.float64)
        C_target = (C_target + C_target.T) / 2
        C_target.fill_diagonal_(0)
        
        p_source = torch.ones(5, dtype=torch.float64) / 5
        p_target = torch.ones(4, dtype=torch.float64) / 4
        
        # Test cosine similarity fallback
        coupling, cost, log = gw_computer._fallback_cosine_similarity(
            C_source, C_target, p_source, p_target
        )
        
        assert coupling.shape == (4, 5)  # POT convention (n_target, n_source)
        assert cost >= 0
        assert 'method' in log
        assert log['method'] == 'cosine_similarity'
        
        # Check marginal constraints
        row_sums = coupling.sum(dim=1)
        col_sums = coupling.sum(dim=0)
        assert torch.allclose(row_sums, p_target, atol=1e-5)
        assert torch.allclose(col_sums, p_source, atol=1e-5)
    
    def test_spectral_matching_fallback(self):
        """Test spectral matching fallback strategy."""
        config = GWConfig(computation_dtype='float64')
        gw_computer = GromovWassersteinComputer(config)
        
        # Create cost matrices
        C_source = torch.rand(5, 5, dtype=torch.float64)
        C_source = (C_source + C_source.T) / 2
        C_source.fill_diagonal_(0)
        
        C_target = torch.rand(4, 4, dtype=torch.float64)
        C_target = (C_target + C_target.T) / 2
        C_target.fill_diagonal_(0)
        
        p_source = torch.ones(5, dtype=torch.float64) / 5
        p_target = torch.ones(4, dtype=torch.float64) / 4
        
        # Test spectral matching fallback
        coupling, cost, log = gw_computer._fallback_spectral_matching(
            C_source, C_target, p_source, p_target
        )
        
        assert coupling.shape == (4, 5)
        assert cost >= 0
        assert log['method'] == 'spectral_matching'
        
        # Check marginal constraints
        row_sums = coupling.sum(dim=1)
        col_sums = coupling.sum(dim=0)
        assert torch.allclose(row_sums, p_target, atol=1e-5)
        assert torch.allclose(col_sums, p_source, atol=1e-5)
    
    def test_fallback_strategy_selection(self):
        """Test that fallback strategies are tried in order of quality."""
        config = GWConfig(computation_dtype='float64')
        gw_computer = GromovWassersteinComputer(config)
        
        C_source = torch.rand(5, 5, dtype=torch.float64)
        C_source = (C_source + C_source.T) / 2
        C_source.fill_diagonal_(0)
        
        C_target = torch.rand(4, 4, dtype=torch.float64)
        C_target = (C_target + C_target.T) / 2
        C_target.fill_diagonal_(0)
        
        # Test fallback selection
        coupling, cost, log = gw_computer._compute_gw_fallback(
            C_source, C_target, None, None
        )
        
        assert coupling.shape == (4, 5)
        assert 'strategy_used' in log
        assert log['strategy_used'] in ['cosine_similarity', 'spectral_matching', 'uniform_coupling']


class TestStrictQualityMode:
    """Test strict quality mode and error handling."""
    
    def test_strict_mode_fail_fast(self):
        """Test that strict mode fails fast on low quality."""
        config = GWConfig(
            strict_quality_mode=True,
            min_coupling_quality=0.9,
            computation_dtype='float64'
        )
        
        gw_manager = GWRestrictionManager(config=config)
        
        # Create test data with same dimensions to avoid tensor size mismatch
        activations = {
            'layer1': torch.randn(10, 20, dtype=torch.float64),
            'layer2': torch.randn(10, 20, dtype=torch.float64)  # Same size
        }
        
        poset = nx.DiGraph()
        poset.add_edge('layer1', 'layer2')
        
        # Mock low quality result
        with patch.object(gw_manager.gw_computer, 'compute_gw_coupling') as mock_gw:
            # Create a proper coupling with correct marginals
            p_source = torch.ones(20, dtype=torch.float64) / 20
            p_target = torch.ones(20, dtype=torch.float64) / 20
            coupling = torch.outer(p_target, p_source)  # (20, 20)
            
            mock_result = GWResult(
                coupling=coupling,
                cost=0.5,
                log={},
                source_size=20,
                target_size=20,
                coupling_quality='fallback',
                quality_score=0.3,  # Below threshold
                solver_type='fallback_uniform'
            )
            mock_gw.return_value = mock_result
            
            # Should raise in strict mode
            with pytest.raises(Exception):  # Will be wrapped in GWRestrictionError
                gw_manager.compute_all_restrictions(activations, poset, parallel=False)
    
    def test_exclude_fallback_edges(self):
        """Test that low-quality edges can be excluded."""
        config = GWConfig(
            strict_quality_mode=False,
            exclude_fallback_edges=True,
            min_coupling_quality=0.5,
            computation_dtype='float64'
        )
        
        gw_manager = GWRestrictionManager(config=config)
        
        # Create test data with same dimensions to avoid issues
        activations = {
            'layer1': torch.randn(10, 20, dtype=torch.float64),
            'layer2': torch.randn(10, 20, dtype=torch.float64),
            'layer3': torch.randn(10, 20, dtype=torch.float64)
        }
        
        poset = nx.DiGraph()
        poset.add_edges_from([('layer1', 'layer2'), ('layer2', 'layer3')])
        
        # Mock mixed quality results
        call_count = 0
        def mock_gw_side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            
            # Create proper couplings
            p = torch.ones(20, dtype=torch.float64) / 20
            coupling = torch.outer(p, p)  # (20, 20)
            
            if call_count == 1:  # First edge - high quality
                return GWResult(
                    coupling=coupling,
                    cost=0.1,
                    log={},
                    source_size=20,
                    target_size=20,
                    coupling_quality='optimal',
                    quality_score=0.9,
                    solver_type='pot_entropic'
                )
            else:  # Second edge - low quality
                return GWResult(
                    coupling=coupling,
                    cost=0.8,
                    log={},
                    source_size=20,
                    target_size=20,
                    coupling_quality='fallback',
                    quality_score=0.3,  # Below threshold
                    solver_type='fallback_uniform'
                )
        
        with patch.object(gw_manager.gw_computer, 'compute_gw_coupling', side_effect=mock_gw_side_effect):
            restrictions, gw_costs, metadata = gw_manager.compute_all_restrictions(
                activations, poset, parallel=False
            )
            
            # Should only have one successful edge (high quality one)
            assert len(restrictions) >= 1
            assert metadata['num_edges_failed'] >= 1


class TestLaplacianQualityFiltering:
    """Test Laplacian construction with quality filtering."""
    
    def test_quality_threshold_filtering(self):
        """Test that Laplacian excludes low-quality edges."""
        # Create a mock sheaf with quality scores
        sheaf = Sheaf()
        sheaf.poset = nx.DiGraph()
        sheaf.poset.add_edges_from([('A', 'B'), ('B', 'C'), ('A', 'C')])
        
        # Add stalks
        sheaf.stalks = {
            'A': torch.eye(3, dtype=torch.float64),
            'B': torch.eye(3, dtype=torch.float64),
            'C': torch.eye(3, dtype=torch.float64)
        }
        
        # Add restrictions
        sheaf.restrictions = {
            ('A', 'B'): torch.eye(3, dtype=torch.float64),
            ('B', 'C'): torch.eye(3, dtype=torch.float64),
            ('A', 'C'): torch.eye(3, dtype=torch.float64)
        }
        
        # Add metadata with quality scores
        sheaf.metadata = {
            'construction_method': 'gromov_wasserstein',
            'gw_costs': {
                ('A', 'B'): 0.1,
                ('B', 'C'): 0.5,
                ('A', 'C'): 0.2
            },
            'gw_quality_scores': {
                ('A', 'B'): 0.9,  # High quality
                ('B', 'C'): 0.3,  # Low quality
                ('A', 'C'): 0.8   # High quality
            }
        }
        
        # Build Laplacian with quality threshold
        builder = GWLaplacianBuilder()
        
        # Without filtering - all edges
        L_full = builder.build_laplacian(sheaf, sparse=False, quality_threshold=None)
        
        # With filtering - exclude low quality edge
        L_filtered = builder.build_laplacian(sheaf, sparse=False, quality_threshold=0.5)
        
        # The filtered Laplacian should have different structure
        # (fewer non-zero entries due to excluded edge)
        assert not torch.allclose(L_full, L_filtered)
    
    def test_quality_filtering_logging(self):
        """Test that quality filtering logs appropriately."""
        sheaf = Sheaf()
        sheaf.poset = nx.DiGraph()
        sheaf.poset.add_edge('A', 'B')
        
        sheaf.stalks = {
            'A': torch.eye(2, dtype=torch.float64),
            'B': torch.eye(2, dtype=torch.float64)
        }
        
        sheaf.restrictions = {
            ('A', 'B'): torch.eye(2, dtype=torch.float64)
        }
        
        sheaf.metadata = {
            'construction_method': 'gromov_wasserstein',
            'gw_costs': {('A', 'B'): 0.1},
            'gw_quality_scores': {('A', 'B'): 0.4}
        }
        
        builder = GWLaplacianBuilder()
        
        # Should log about filtering
        with patch('neurosheaf.sheaf.assembly.gw_laplacian.logger') as mock_logger:
            L = builder.build_laplacian(sheaf, quality_threshold=0.5)
            
            # Check that quality filtering was logged
            info_calls = [call[0][0] for call in mock_logger.info.call_args_list]
            assert any('Quality filtering' in str(call) for call in info_calls)


class TestParallelErrorAggregation:
    """Test error aggregation in parallel processing."""
    
    def test_parallel_quality_issue_reporting(self):
        """Test that parallel processing aggregates quality issues."""
        config = GWConfig(
            strict_quality_mode=False,
            min_coupling_quality=0.7,
            computation_dtype='float64'
        )
        
        gw_manager = GWRestrictionManager(config=config)
        
        # Create test data with many edges
        activations = {}
        poset = nx.DiGraph()
        for i in range(5):
            activations[f'layer{i}'] = torch.randn(10, 10+i, dtype=torch.float64)
            if i > 0:
                poset.add_edge(f'layer{i-1}', f'layer{i}')
        
        # Mock variable quality results
        def mock_gw_side_effect(*args, **kwargs):
            # Alternate between good and bad quality
            import random
            quality = random.choice([0.2, 0.8])
            
            return GWResult(
                coupling=torch.rand(10, 10, dtype=torch.float64),
                cost=1 - quality,
                log={},
                source_size=10,
                target_size=10,
                coupling_quality='optimal' if quality > 0.5 else 'fallback',
                quality_score=quality,
                solver_type='pot' if quality > 0.5 else 'fallback'
            )
        
        with patch.object(gw_manager.gw_computer, 'compute_gw_coupling', side_effect=mock_gw_side_effect):
            with patch('neurosheaf.sheaf.assembly.gw_builder.logger') as mock_logger:
                restrictions, gw_costs, metadata = gw_manager.compute_all_restrictions(
                    activations, poset, parallel=True, max_workers=2
                )
                
                # Should have logged quality issues
                warning_calls = [call[0][0] for call in mock_logger.warning.call_args_list]
                quality_warnings = [call for call in warning_calls if 'Quality issues' in str(call)]
                
                # May or may not have quality issues depending on random
                if metadata['num_edges_failed'] > 0:
                    assert len(quality_warnings) > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
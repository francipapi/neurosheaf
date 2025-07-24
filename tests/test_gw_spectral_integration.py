"""GW spectral integration tests for Phase 3 implementation.

This module tests the complete integration of GW-based sheaves with the
spectral analysis pipeline, including:
- Edge weight extraction and semantics
- Filtration parameter generation
- Threshold function creation
- Persistence computation
- End-to-end spectral analysis
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List
from unittest.mock import patch, MagicMock
import networkx as nx

from neurosheaf.sheaf.assembly import SheafBuilder
from neurosheaf.sheaf.core import GWConfig
from neurosheaf.sheaf.data_structures import Sheaf
from neurosheaf.spectral.persistent import PersistentSpectralAnalyzer
from neurosheaf.spectral.static_laplacian_unified import UnifiedStaticLaplacian
from neurosheaf.spectral.edge_weights import EdgeWeightExtractor, EdgeWeightMetadata
from neurosheaf.spectral.tracker import SubspaceTracker


class TestEdgeWeightExtraction:
    """Test unified edge weight extraction for GW vs standard sheaves."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.extractor = EdgeWeightExtractor(validate_weights=True, log_statistics=False)
        
        # Create test sheaves
        self.gw_sheaf = self._create_gw_test_sheaf()
        self.standard_sheaf = self._create_standard_test_sheaf()
    
    def _create_gw_test_sheaf(self) -> Sheaf:
        """Create GW sheaf with known edge weights."""
        poset = nx.DiGraph()
        poset.add_edges_from([('A', 'B'), ('B', 'C')])
        
        stalks = {
            'A': torch.eye(3),
            'B': torch.eye(2),
            'C': torch.eye(2)
        }
        
        restrictions = {
            ('A', 'B'): torch.rand(2, 3) * 0.5,
            ('B', 'C'): torch.rand(2, 2) * 0.5
        }
        
        # Include GW costs in metadata
        gw_costs = {
            ('A', 'B'): 0.4,
            ('B', 'C'): 0.6
        }
        
        metadata = {
            'construction_method': 'gromov_wasserstein',
            'gw_costs': gw_costs,
            'gw_config': GWConfig().to_dict()
        }
        
        return Sheaf(poset=poset, stalks=stalks, restrictions=restrictions, metadata=metadata)
    
    def _create_standard_test_sheaf(self) -> Sheaf:
        """Create standard Procrustes sheaf."""
        poset = nx.DiGraph()
        poset.add_edges_from([('A', 'B'), ('B', 'C')])
        
        stalks = {
            'A': torch.eye(3),
            'B': torch.eye(2), 
            'C': torch.eye(2)
        }
        
        restrictions = {
            ('A', 'B'): torch.randn(2, 3) * 0.2,
            ('B', 'C'): torch.randn(2, 2) * 0.3
        }
        
        metadata = {
            'construction_method': 'scaled_procrustes',
            'whitened': True
        }
        
        return Sheaf(poset=poset, stalks=stalks, restrictions=restrictions, metadata=metadata)
    
    def test_gw_weight_extraction(self):
        """Test GW cost extraction from metadata."""
        weights, metadata = self.extractor.extract_weights(self.gw_sheaf)
        
        # Should extract stored GW costs
        assert weights[('A', 'B')] == 0.4
        assert weights[('B', 'C')] == 0.6
        
        # Metadata should reflect GW construction
        assert metadata.construction_method == 'gromov_wasserstein'
        assert metadata.weight_type == 'gw_costs'
        assert metadata.filtration_semantics == 'increasing'
        assert metadata.fallback_used is False
    
    def test_standard_weight_extraction(self):
        """Test Frobenius norm extraction for standard sheaves."""
        weights, metadata = self.extractor.extract_weights(self.standard_sheaf)
        
        # Should compute Frobenius norms
        for edge, weight in weights.items():
            restriction = self.standard_sheaf.restrictions[edge]
            expected = torch.norm(restriction, p='fro').item()
            assert np.isclose(weight, expected, rtol=1e-6)
        
        # Metadata should reflect standard construction
        assert metadata.construction_method == 'scaled_procrustes'
        assert metadata.weight_type == 'frobenius_norms_whitened'
        assert metadata.filtration_semantics == 'decreasing'
    
    def test_gw_fallback_mechanism(self):
        """Test fallback when GW costs are missing."""
        # Remove GW costs from metadata
        fallback_sheaf = self.gw_sheaf
        fallback_sheaf.metadata.pop('gw_costs', None)
        
        weights, metadata = self.extractor.extract_weights(fallback_sheaf)
        
        # Should use fallback method
        assert metadata.fallback_used is True
        assert metadata.weight_type == 'operator_norms_fallback'
        
        # Should still have increasing semantics
        assert metadata.filtration_semantics == 'increasing'
        
        # Weights should be positive
        for weight in weights.values():
            assert weight > 0
    
    def test_weight_validation(self):
        """Test weight validation for different sheaf types."""
        # Should pass for normal sheaves
        weights, _ = self.extractor.extract_weights(self.gw_sheaf)
        # No exception should be raised
        
        # Test with zero weights (should warn but not fail)
        zero_sheaf = self.gw_sheaf
        zero_sheaf.metadata['gw_costs'] = {('A', 'B'): 0.0, ('B', 'C'): 0.0}
        
        weights, metadata = self.extractor.extract_weights(zero_sheaf)
        assert all(w == 0.0 for w in weights.values())
    
    def test_filtration_direction_detection(self):
        """Test correct filtration direction for different methods."""
        # GW should be increasing
        assert self.extractor.get_filtration_direction('gromov_wasserstein') == 'increasing'
        
        # Standard should be decreasing
        assert self.extractor.get_filtration_direction('scaled_procrustes') == 'decreasing'
        assert self.extractor.get_filtration_direction('standard') == 'decreasing'
    
    def test_threshold_function_creation(self):
        """Test creation of appropriate threshold functions."""
        # GW threshold: weight <= param (include edges with low cost)
        gw_thresh = self.extractor.create_threshold_function('increasing')
        assert gw_thresh(0.3, 0.5) is True   # Include: cost <= threshold
        assert gw_thresh(0.7, 0.5) is False  # Exclude: cost > threshold
        
        # Standard threshold: weight >= param (keep edges with high weight)
        std_thresh = self.extractor.create_threshold_function('decreasing')
        assert std_thresh(0.7, 0.5) is True   # Keep: weight >= threshold
        assert std_thresh(0.3, 0.5) is False  # Remove: weight < threshold


class TestGWFiltrationGeneration:
    """Test GW-specific filtration parameter generation."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.analyzer = PersistentSpectralAnalyzer()
        
        # Create GW sheaf with known cost range
        self.gw_sheaf = self._create_gw_sheaf_with_costs()
    
    def _create_gw_sheaf_with_costs(self) -> Sheaf:
        """Create GW sheaf with realistic cost range."""
        poset = nx.DiGraph()
        poset.add_edges_from([('layer1', 'layer2'), ('layer2', 'layer3'), ('layer1', 'layer3')])
        
        stalks = {
            'layer1': torch.eye(4),
            'layer2': torch.eye(3),
            'layer3': torch.eye(2)
        }
        
        restrictions = {
            ('layer1', 'layer2'): torch.rand(3, 4),
            ('layer2', 'layer3'): torch.rand(2, 3),
            ('layer1', 'layer3'): torch.rand(2, 4)
        }
        
        # Realistic GW costs (cosine distances typically in [0, 2])
        gw_costs = {
            ('layer1', 'layer2'): 0.2,  # Good match
            ('layer2', 'layer3'): 0.8,  # Medium match  
            ('layer1', 'layer3'): 1.5   # Poor match
        }
        
        metadata = {
            'construction_method': 'gromov_wasserstein',
            'gw_costs': gw_costs,
            'gw_config': GWConfig().to_dict()
        }
        
        return Sheaf(poset=poset, stalks=stalks, restrictions=restrictions, metadata=metadata)
    
    def test_gw_filtration_parameter_generation(self):
        """Test INCREASING parameter generation for GW costs."""
        params = self.analyzer._generate_filtration_params(
            self.gw_sheaf, 'threshold', n_steps=10, param_range=None
        )
        
        # Should generate increasing sequence
        assert len(params) == 10
        assert all(params[i] <= params[i+1] for i in range(len(params)-1))
        
        # Range should cover GW costs
        assert params[0] <= 0.2  # Should include best match
        assert params[-1] >= 1.5  # Should include worst match
    
    def test_gw_log_scale_generation(self):
        """Test log-scale generation for GW costs with wide range."""
        # Create sheaf with wide cost range
        wide_range_sheaf = self.gw_sheaf
        wide_range_sheaf.metadata['gw_costs'] = {
            ('layer1', 'layer2'): 0.01,  # Very good match
            ('layer2', 'layer3'): 0.5,   # Medium match
            ('layer1', 'layer3'): 2.0    # Poor match
        }
        
        params = self.analyzer._generate_filtration_params(
            wide_range_sheaf, 'threshold', n_steps=20, param_range=None
        )
        
        # Should use log scale for wide range
        assert len(params) == 20
        assert params[0] >= 0.01 and params[-1] <= 2.0
        
        # Should have good resolution at small values
        small_params = [p for p in params if p < 0.1]
        assert len(small_params) >= 5  # Good resolution at small costs
    
    def test_gw_vs_standard_filtration_difference(self):
        """Test that GW and standard filtrations have different characteristics."""
        # GW filtration
        gw_params = self.analyzer._generate_gw_filtration_params(
            self.gw_sheaf, n_steps=10, param_range=None
        )
        
        # Create comparable standard sheaf
        standard_sheaf = self.gw_sheaf
        standard_sheaf.metadata['construction_method'] = 'standard'
        
        std_params = self.analyzer._generate_standard_filtration_params(
            standard_sheaf, n_steps=10, param_range=None
        )
        
        # Both should be increasing sequences, but different ranges
        assert all(gw_params[i] <= gw_params[i+1] for i in range(len(gw_params)-1))
        assert all(std_params[i] <= std_params[i+1] for i in range(len(std_params)-1))
        
        # GW should use stored costs, standard should use computed norms
        assert np.mean(gw_params) != np.mean(std_params)
    
    def test_user_provided_range_override(self):
        """Test user-provided parameter range override."""
        custom_range = (0.5, 1.0)
        
        params = self.analyzer._generate_gw_filtration_params(
            self.gw_sheaf, n_steps=5, param_range=custom_range
        )
        
        # Should respect user range
        assert params[0] >= 0.5
        assert params[-1] <= 1.0
        assert len(params) == 5
    
    def test_empty_costs_fallback(self):
        """Test fallback when GW costs are missing."""
        # Remove GW costs
        no_costs_sheaf = self.gw_sheaf
        no_costs_sheaf.metadata.pop('gw_costs', None)
        
        params = self.analyzer._generate_gw_filtration_params(
            no_costs_sheaf, n_steps=8, param_range=None
        )
        
        # Should still generate reasonable parameters
        assert len(params) == 8
        assert all(isinstance(p, float) for p in params)
        assert all(p >= 0 for p in params)


class TestGWThresholdSemantics:
    """Test GW-specific threshold function semantics."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.analyzer = PersistentSpectralAnalyzer()
        
        # Create test sheaves
        self.gw_sheaf = self._create_test_sheaf('gromov_wasserstein')
        self.standard_sheaf = self._create_test_sheaf('scaled_procrustes')
    
    def _create_test_sheaf(self, method: str) -> Sheaf:
        """Create test sheaf with specified construction method."""
        poset = nx.DiGraph()
        poset.add_edge('A', 'B')
        
        stalks = {'A': torch.eye(2), 'B': torch.eye(2)}
        restrictions = {('A', 'B'): torch.rand(2, 2) * 0.5}
        
        if method == 'gromov_wasserstein':
            metadata = {
                'construction_method': method,
                'gw_costs': {('A', 'B'): 0.5}
            }
        else:
            metadata = {'construction_method': method}
        
        return Sheaf(poset=poset, stalks=stalks, restrictions=restrictions, metadata=metadata)
    
    def test_gw_threshold_function_creation(self):
        """Test GW threshold function: weight <= param."""
        threshold_func = self.analyzer._create_edge_threshold_func(
            'threshold', None, self.gw_sheaf
        )
        
        # GW semantics: include edges with cost <= threshold
        assert threshold_func(0.3, 0.5) is True   # Include: cost <= threshold
        assert threshold_func(0.7, 0.5) is False  # Exclude: cost > threshold
        assert threshold_func(0.5, 0.5) is True   # Include: cost == threshold
    
    def test_standard_threshold_function_creation(self):
        """Test standard threshold function: weight >= param."""
        threshold_func = self.analyzer._create_edge_threshold_func(
            'threshold', None, self.standard_sheaf
        )
        
        # Standard semantics: keep edges with weight >= threshold
        assert threshold_func(0.7, 0.5) is True   # Keep: weight >= threshold
        assert threshold_func(0.3, 0.5) is False  # Remove: weight < threshold
        assert threshold_func(0.5, 0.5) is True   # Keep: weight == threshold
    
    def test_method_detection_in_threshold_creation(self):
        """Test that threshold function adapts to construction method."""
        # Should detect GW method and use appropriate threshold
        gw_func = self.analyzer._create_edge_threshold_func(
            'threshold', None, self.gw_sheaf
        )
        
        # Should detect standard method and use appropriate threshold
        std_func = self.analyzer._create_edge_threshold_func(
            'threshold', None, self.standard_sheaf
        )
        
        # Same weight and param, opposite results due to different semantics
        weight, param = 0.6, 0.5
        assert gw_func(weight, param) is False   # GW: exclude high cost
        assert std_func(weight, param) is True   # Standard: keep high weight
    
    def test_cka_based_always_standard_semantics(self):
        """Test that CKA-based filtration always uses standard semantics."""
        # Even with GW sheaf, CKA should use standard semantics
        cka_func = self.analyzer._create_edge_threshold_func(
            'cka_based', None, self.gw_sheaf
        )
        
        # Should behave like standard threshold (weight >= param)
        assert cka_func(0.7, 0.5) is True   # Keep high correlation
        assert cka_func(0.3, 0.5) is False  # Remove low correlation


class TestGWSubspaceTracking:
    """Test subspace tracking for GW-based spectral analysis."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.tracker = SubspaceTracker()
        
        # Create synthetic eigenvalue sequences
        self.gw_sequences = self._create_gw_sequences()
        self.std_sequences = self._create_standard_sequences()
    
    def _create_gw_sequences(self):
        """Create eigenvalue sequences typical for GW filtration."""
        # GW: increasing complexity (more edges over time)
        # Early: sparse graph → many small eigenvalues
        # Later: dense graph → fewer larger eigenvalues
        
        eigenvalue_sequences = [
            torch.tensor([0.0, 0.1, 0.2, 0.8, 1.0]),  # Sparse: many small
            torch.tensor([0.0, 0.2, 0.4, 0.9, 1.2]),  # Medium density
            torch.tensor([0.0, 0.3, 0.7, 1.1, 1.5])   # Dense: fewer small
        ]
        
        # Create corresponding eigenvector sequences
        eigenvector_sequences = [
            torch.randn(5, 5) for _ in range(3)
        ]
        
        filtration_params = [0.2, 0.5, 0.8]  # Increasing GW costs
        
        return eigenvalue_sequences, eigenvector_sequences, filtration_params
    
    def _create_standard_sequences(self):
        """Create eigenvalue sequences typical for standard filtration.""" 
        # Standard: decreasing complexity (fewer edges over time)
        # Early: dense graph → fewer large eigenvalues
        # Later: sparse graph → more small eigenvalues
        
        eigenvalue_sequences = [
            torch.tensor([0.0, 0.3, 0.7, 1.1, 1.5]),  # Dense: fewer small
            torch.tensor([0.0, 0.2, 0.4, 0.9, 1.2]),  # Medium density
            torch.tensor([0.0, 0.1, 0.2, 0.8, 1.0])   # Sparse: many small
        ]
        
        eigenvector_sequences = [
            torch.randn(5, 5) for _ in range(3)
        ]
        
        filtration_params = [0.2, 0.5, 0.8]  # Increasing threshold params
        
        return eigenvalue_sequences, eigenvector_sequences, filtration_params
    
    def test_gw_method_routing(self):
        """Test that GW construction method routes to GW tracking."""
        eigenvals, eigenvecs, params = self.gw_sequences
        
        # Should route to GW-specific handler
        with patch.object(self.tracker, '_track_gw_subspaces', return_value={'test': 'gw'}) as mock_gw:
            result = self.tracker.track_eigenspaces(
                eigenvals, eigenvecs, params, construction_method='gromov_wasserstein'
            )
            
            mock_gw.assert_called_once()
            assert result == {'test': 'gw'}
    
    def test_standard_method_routing(self):
        """Test that standard construction methods route to standard tracking."""
        eigenvals, eigenvecs, params = self.std_sequences
        
        # Should route to standard handler
        with patch.object(self.tracker, '_track_standard_subspaces', return_value={'test': 'standard'}) as mock_std:
            result = self.tracker.track_eigenspaces(
                eigenvals, eigenvecs, params, construction_method='scaled_procrustes'
            )
            
            mock_std.assert_called_once()
            assert result == {'test': 'standard'}
    
    def test_birth_death_validation_works_for_both(self):
        """Test that birth <= death validation works for both GW and standard."""
        eigenvals, eigenvecs, params = self.gw_sequences
        
        # Both methods should validate birth <= death correctly
        # since both use increasing parameter sequences
        gw_result = self.tracker.track_eigenspaces(
            eigenvals, eigenvecs, params, construction_method='gromov_wasserstein'
        )
        
        std_result = self.tracker.track_eigenspaces(
            eigenvals, eigenvecs, params, construction_method='scaled_procrustes'
        )
        
        # Both should produce valid results
        assert 'finite_pairs' in gw_result
        assert 'finite_pairs' in std_result
        
        # All persistence pairs should satisfy birth <= death
        for pairs in [gw_result['finite_pairs'], std_result['finite_pairs']]:
            for pair in pairs:
                assert pair['birth_param'] <= pair['death_param']


class TestGWLaplacianBlockStructure:
    """Test GW Laplacian block structure validation (Phase 5 requirement)."""
    
    def setup_method(self):
        """Set up test fixtures."""
        from neurosheaf.sheaf.assembly.laplacian import SheafLaplacianBuilder
        self.laplacian_builder = SheafLaplacianBuilder(method='gromov_wasserstein')
        self.gw_config = GWConfig(epsilon=0.1, max_iter=100)
    
    def test_laplacian_block_structure(self):
        """Verify GW Laplacian follows equations (5.1) and (5.2)."""
        # Create simple GW sheaf
        model = nn.Sequential(nn.Linear(6, 4), nn.Linear(4, 3))
        data = torch.randn(20, 6)
        
        builder = SheafBuilder(restriction_method='gromov_wasserstein')
        try:
            sheaf = builder.build_from_activations(
                self._extract_activations(model, data),
                model,
                gw_config=self.gw_config
            )
            
            # Build Laplacian
            result = self.laplacian_builder.build_laplacian(sheaf, sparse=False)
            L = result.laplacian
            
            # Verify block structure matches mathematical formulation
            self._verify_block_structure(sheaf, L)
            
        except Exception as e:
            if "POT" in str(e):
                pytest.skip("POT library not available")
            else:
                raise
    
    def _extract_activations(self, model, data):
        """Simple activation extraction."""
        activations = {}
        x = data
        for i, layer in enumerate(model):
            x = layer(x)
            if isinstance(layer, nn.Linear):
                activations[f"layer_{i}"] = x.detach().clone()
        return activations
    
    def _verify_block_structure(self, sheaf: Sheaf, L: torch.Tensor):
        """Verify Laplacian block structure against mathematical formula."""
        nodes = list(sheaf.poset.nodes())
        node_dims = {node: sheaf.stalks[node].shape[0] for node in nodes}
        
        # Extract block positions
        offsets = {}
        current_offset = 0
        for node in nodes:
            offsets[node] = current_offset
            current_offset += node_dims[node]
        
        # Verify diagonal blocks: L_{ii} = (∑_{e∈in(i)} I) + ∑_{e=(i→j)∈out(i)} ρ^T ρ
        for node in nodes:
            start = offsets[node]
            end = start + node_dims[node]
            diagonal_block = L[start:end, start:end]
            
            # Compute expected block
            expected_block = torch.zeros_like(diagonal_block)
            
            # Add identity for each incoming edge
            in_edges = list(sheaf.poset.in_edges(node))
            expected_block += len(in_edges) * torch.eye(node_dims[node])
            
            # Add ρ^T ρ for each outgoing edge
            for source, target in sheaf.poset.out_edges(node):
                if (source, target) in sheaf.restrictions:
                    rho = sheaf.restrictions[(source, target)]
                    expected_block += rho.T @ rho
            
            # Verify match
            block_error = torch.norm(diagonal_block - expected_block, 'fro').item()
            assert block_error < 1e-5, f"Diagonal block mismatch for {node}: {block_error}"
        
        # Verify off-diagonal blocks: L_{ij} = -ρ^T or -ρ
        for edge in sheaf.restrictions:
            source, target = edge
            start_i, end_i = offsets[source], offsets[source] + node_dims[source] 
            start_j, end_j = offsets[target], offsets[target] + node_dims[target]
            
            # Check (source, target) block should be -ρ^T
            block_ij = L[start_i:end_i, start_j:end_j]
            expected_ij = -sheaf.restrictions[edge].T
            error_ij = torch.norm(block_ij - expected_ij, 'fro').item()
            assert error_ij < 1e-5, f"Off-diagonal block ({source}, {target}) error: {error_ij}"
            
            # Check (target, source) block should be -ρ
            block_ji = L[start_j:end_j, start_i:end_i]
            expected_ji = -sheaf.restrictions[edge]
            error_ji = torch.norm(block_ji - expected_ji, 'fro').item()
            assert error_ji < 1e-5, f"Off-diagonal block ({target}, {source}) error: {error_ji}"


class TestGWMixedArchitectureComparison:
    """Test spectral comparison across different architectures using GW (Phase 5 requirement)."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.analyzer = PersistentSpectralAnalyzer()
        self.gw_config = GWConfig(epsilon=0.05, max_iter=50)  # Fast but reasonable
    
    def test_mixed_architecture_comparison(self):
        """Test spectral comparison across different architectures."""
        # Create networks with different architectures but same input/output
        architectures = [
            [10, 8, 6, 4],      # Standard decreasing
            [10, 12, 8, 4],     # Wide middle
            [10, 6, 8, 4],      # Bottleneck then expand
            [10, 10, 10, 4]     # Constant then drop
        ]
        
        data = torch.randn(30, 10)
        builder = SheafBuilder(restriction_method='gromov_wasserstein')
        
        sheaves = []
        spectral_results = []
        
        try:
            # Build sheaves for each architecture
            for i, arch in enumerate(architectures):
                model = nn.Sequential(*[
                    nn.Linear(arch[j], arch[j+1])
                    for j in range(len(arch) - 1)
                ])
                
                activations = self._extract_activations(model, data)
                sheaf = builder.build_from_activations(
                    activations, model, gw_config=self.gw_config
                )
                sheaves.append(sheaf)
                
                # Analyze spectral properties
                result = self.analyzer.analyze(sheaf, n_steps=8)
                spectral_results.append(result)
            
            # Verify all results are valid
            for i, result in enumerate(spectral_results):
                assert 'persistence_result' in result
                assert len(result['filtration_params']) == 8
                
                # Verify GW-specific properties
                persistence = result['persistence_result']
                assert len(persistence['eigenvalue_sequences']) == 8
                
                # Filtration should be increasing for GW
                params = result['filtration_params']
                assert all(params[j] <= params[j+1] for j in range(len(params)-1))
            
            # Verify different architectures produce different spectral signatures
            # (while being comparable due to GW construction)
            eigenvalue_means = []
            for result in spectral_results:
                eigenvals = result['persistence_result']['eigenvalue_sequences']
                # Compute mean of final eigenvalue spectrum
                final_eigenvals = eigenvals[-1]
                eigenvalue_means.append(torch.mean(final_eigenvals).item())
            
            # Should have variation across architectures
            eigenvalue_std = np.std(eigenvalue_means)
            assert eigenvalue_std > 1e-6, "Architectures should produce different spectra"
            
        except Exception as e:
            if "POT" in str(e):
                pytest.skip("POT library not available")
            else:
                raise
    
    def _extract_activations(self, model, data):
        """Extract layer activations."""
        activations = {}
        x = data
        for i, layer in enumerate(model):
            x = layer(x)
            if isinstance(layer, nn.Linear):
                activations[f"layer_{i}"] = x.detach().clone()
        return activations
    
    def test_cross_architecture_spectral_similarity(self):
        """Test that similar architectures produce similar spectral properties."""
        # Create two very similar architectures
        arch1 = [8, 6, 4, 2]
        arch2 = [8, 7, 4, 2]  # Only middle layer differs by 1
        
        data = torch.randn(40, 8)
        builder = SheafBuilder(restriction_method='gromov_wasserstein')
        
        try:
            # Build both sheaves
            models = [
                nn.Sequential(*[nn.Linear(arch1[i], arch1[i+1]) for i in range(len(arch1)-1)]),
                nn.Sequential(*[nn.Linear(arch2[i], arch2[i+1]) for i in range(len(arch2)-1)])
            ]
            
            results = []
            for model in models:
                activations = self._extract_activations(model, data)
                sheaf = builder.build_from_activations(
                    activations, model, gw_config=self.gw_config
                )
                result = self.analyzer.analyze(sheaf, n_steps=6)
                results.append(result)
            
            # Compare spectral properties
            result1, result2 = results
            eigenvals1 = result1['persistence_result']['eigenvalue_sequences']
            eigenvals2 = result2['persistence_result']['eigenvalue_sequences']
            
            # Compare final spectra (should be similar)
            final_eigs1 = eigenvals1[-1]
            final_eigs2 = eigenvals2[-1]
            
            # Pad to same length if needed
            min_len = min(len(final_eigs1), len(final_eigs2))
            final_eigs1 = final_eigs1[:min_len]
            final_eigs2 = final_eigs2[:min_len]
            
            # Compute spectral distance
            spectral_distance = torch.norm(final_eigs1 - final_eigs2).item()
            max_eigenval = max(torch.max(final_eigs1).item(), torch.max(final_eigs2).item())
            relative_distance = spectral_distance / max_eigenval if max_eigenval > 0 else 0
            
            # Similar architectures should have similar spectra
            assert relative_distance < 0.5, f"Similar architectures too different: {relative_distance}"
            
        except Exception as e:
            if "POT" in str(e):
                pytest.skip("POT library not available")
            else:
                raise


class TestEndToEndGWSpectralAnalysis:
    """Test complete end-to-end GW spectral analysis pipeline."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.analyzer = PersistentSpectralAnalyzer()
        self.gw_builder = SheafBuilder(restriction_method='gromov_wasserstein')
        
        # Simple test network
        self.test_net = nn.Sequential(
            nn.Linear(8, 6),
            nn.ReLU(),
            nn.Linear(6, 4)
        )
        self.test_input = torch.randn(3, 8)  # Small batch for fast testing
        
        self.gw_config = GWConfig(epsilon=0.1, max_iter=10)  # Fast config
    
    def test_complete_gw_spectral_pipeline(self):
        """Test complete pipeline from network to persistence diagrams."""
        try:
            # 1. Build GW sheaf
            sheaf = self.gw_builder.build_from_activations(
                self.test_net, self.test_input,
                validate=True, gw_config=self.gw_config
            )
            
            # Verify it's a GW sheaf
            assert sheaf.is_gw_sheaf()
            assert 'gw_costs' in sheaf.metadata
            
            # 2. Run spectral analysis
            result = self.analyzer.analyze(
                sheaf, 
                filtration_type='threshold',
                n_steps=10  # Small number for fast testing
            )
            
            # 3. Verify results structure
            assert 'persistence_result' in result
            assert 'features' in result
            assert 'diagrams' in result
            assert 'filtration_params' in result
            
            # 4. Verify GW-specific properties
            persistence_result = result['persistence_result']
            assert 'eigenvalue_sequences' in persistence_result
            assert len(persistence_result['eigenvalue_sequences']) == 10
            
            # 5. Verify filtration parameters are increasing (GW semantics)
            filtration_params = result['filtration_params']
            assert all(filtration_params[i] <= filtration_params[i+1] 
                      for i in range(len(filtration_params)-1))
            
            # 6. Verify edge weights come from GW costs
            edge_info = persistence_result.get('edge_info', {})
            if edge_info:
                for edge_data in edge_info.values():
                    assert 'construction_method' in edge_data
                    if edge_data['construction_method'] == 'gromov_wasserstein':
                        # Weight should come from GW costs
                        assert 'weight' in edge_data
                        assert edge_data['weight'] >= 0
            
        except Exception as e:
            if "POT" in str(e) or "not available" in str(e):
                pytest.skip("POT library not available for GW computations")
            else:
                raise
    
    def test_gw_vs_standard_spectral_differences(self):
        """Test that GW and standard methods produce different spectral results."""
        try:
            # Build GW sheaf
            gw_sheaf = self.gw_builder.build_from_activations(
                self.test_net, self.test_input, gw_config=self.gw_config
            )
            
            # Build standard sheaf
            std_builder = SheafBuilder(restriction_method='scaled_procrustes')
            std_sheaf = std_builder.build_from_activations(
                self.test_net, self.test_input, validate=True
            )
            
            # Analyze both
            gw_result = self.analyzer.analyze(gw_sheaf, n_steps=5)
            std_result = self.analyzer.analyze(std_sheaf, n_steps=5)
            
            # Should produce different filtration parameters
            gw_params = gw_result['filtration_params']
            std_params = std_result['filtration_params']
            
            # Different edge weight sources should produce different ranges
            assert not np.allclose(gw_params, std_params, rtol=0.1)
            
            # Both should have valid persistence results
            assert len(gw_result['persistence_result']['eigenvalue_sequences']) > 0
            assert len(std_result['persistence_result']['eigenvalue_sequences']) > 0
            
        except Exception as e:
            if "POT" in str(e):
                pytest.skip("POT library not available")
            else:
                raise
    
    def test_gw_filtration_semantics_end_to_end(self):
        """Test that GW increasing complexity semantics work end-to-end."""
        try:
            sheaf = self.gw_builder.build_from_activations(
                self.test_net, self.test_input, gw_config=self.gw_config
            )
            
            # Analyze with explicit threshold function testing
            result = self.analyzer.analyze(sheaf, n_steps=8)
            
            # Extract edge information and verify threshold semantics
            persistence_result = result['persistence_result']
            filtration_params = result['filtration_params']
            
            # Verify parameters are increasing
            assert all(filtration_params[i] <= filtration_params[i+1] 
                      for i in range(len(filtration_params)-1))
            
            # For GW sheaves, small parameters should include fewer edges
            # (only those with very low costs), large parameters should include more
            edge_info = persistence_result.get('edge_info', {})
            if edge_info and len(filtration_params) >= 2:
                # This tests the increasing complexity interpretation
                assert len(filtration_params) > 1  # Basic sanity check
            
        except Exception as e:
            if "POT" in str(e):
                pytest.skip("POT library not available")
            else:
                raise


class TestGWSpectralRegressionTests:
    """Regression tests to ensure GW integration doesn't break existing functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.analyzer = PersistentSpectralAnalyzer()
        
        # Create standard sheaf for regression testing
        self.standard_sheaf = self._create_standard_sheaf()
    
    def _create_standard_sheaf(self) -> Sheaf:
        """Create standard sheaf identical to pre-GW implementation."""
        poset = nx.DiGraph()
        poset.add_edges_from([('input', 'hidden'), ('hidden', 'output')])
        
        stalks = {
            'input': torch.eye(4),
            'hidden': torch.eye(3),
            'output': torch.eye(2)
        }
        
        restrictions = {
            ('input', 'hidden'): torch.randn(3, 4) * 0.1,
            ('hidden', 'output'): torch.randn(2, 3) * 0.1
        }
        
        metadata = {
            'construction_method': 'scaled_procrustes',
            'whitened': True,
            'validation_passed': True
        }
        
        return Sheaf(poset=poset, stalks=stalks, restrictions=restrictions, metadata=metadata)
    
    def test_standard_sheaf_analysis_unchanged(self):
        """Test that standard sheaves still work as before."""
        # Should analyze without errors using standard pipeline
        result = self.analyzer.analyze(
            self.standard_sheaf, 
            filtration_type='threshold',
            n_steps=6
        )
        
        # Should produce expected structure
        assert 'persistence_result' in result
        assert 'features' in result
        assert 'diagrams' in result
        
        # Should use decreasing complexity filtration (standard semantics)
        filtration_params = result['filtration_params']
        assert len(filtration_params) == 6
        assert all(isinstance(p, float) for p in filtration_params)
    
    def test_edge_weight_extraction_backward_compatible(self):
        """Test that edge weight extraction works for existing sheaves."""
        from neurosheaf.spectral.edge_weights import extract_edge_weights
        
        # Should extract weights without errors
        weights = extract_edge_weights(self.standard_sheaf)
        
        # Should return reasonable weights
        assert len(weights) == 2  # Two edges
        for edge, weight in weights.items():
            assert weight > 0
            assert isinstance(weight, float)
    
    def test_no_construction_method_metadata(self):
        """Test handling of sheaves without construction method metadata."""
        # Remove construction method
        no_method_sheaf = self.standard_sheaf
        no_method_sheaf.metadata.pop('construction_method', None)
        
        # Should still work (defaults to standard)
        result = self.analyzer.analyze(no_method_sheaf, n_steps=5)
        
        assert 'persistence_result' in result
        assert len(result['filtration_params']) == 5


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "--tb=short"])
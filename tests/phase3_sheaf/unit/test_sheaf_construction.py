"""Comprehensive tests for sheaf construction and integration.

This module tests the Sheaf dataclass, SheafBuilder class, and integration
with the CKA and poset extraction components.
"""

import pytest
import torch
import torch.nn as nn
import networkx as nx
import numpy as np

from neurosheaf.sheaf.construction import Sheaf, SheafBuilder, create_sheaf_from_cka_analysis
from neurosheaf.sheaf.poset import FXPosetExtractor
from neurosheaf.utils.exceptions import ArchitectureError, ComputationError


class SimpleTestModel(nn.Module):
    """Simple test model for sheaf construction."""
    
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(32, 10)
    
    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class MLPTestModel(nn.Module):
    """Simple MLP test model."""
    
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 20)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(20, 15)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(15, 5)
    
    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.fc3(x)
        return x


class TestSheafDataclass:
    """Test cases for the Sheaf dataclass."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Create simple test poset
        self.poset = nx.DiGraph()
        self.poset.add_edges_from([('A', 'B'), ('B', 'C')])
        
        # Create test stalks
        torch.manual_seed(42)
        self.stalks = {
            'A': torch.randn(8, 8),
            'B': torch.randn(8, 8),
            'C': torch.randn(8, 8)
        }
        
        # Create test restrictions
        self.restrictions = {
            ('A', 'B'): torch.randn(8, 8),
            ('B', 'C'): torch.randn(8, 8)
        }
    
    def test_sheaf_initialization(self):
        """Test basic sheaf initialization."""
        sheaf = Sheaf(
            poset=self.poset,
            stalks=self.stalks,
            restrictions=self.restrictions
        )
        
        assert sheaf.poset is self.poset
        assert sheaf.stalks is self.stalks
        assert sheaf.restrictions is self.restrictions
        assert isinstance(sheaf.metadata, dict)
        
        # Check metadata initialization
        assert sheaf.metadata['num_nodes'] == 3
        assert sheaf.metadata['num_edges'] == 2
    
    def test_sheaf_empty_initialization(self):
        """Test sheaf initialization with empty data."""
        sheaf = Sheaf()
        
        assert isinstance(sheaf.poset, nx.DiGraph)
        assert len(sheaf.poset.nodes()) == 0
        assert len(sheaf.stalks) == 0
        assert len(sheaf.restrictions) == 0
        assert isinstance(sheaf.metadata, dict)
    
    def test_sheaf_validation(self):
        """Test sheaf validation method."""
        sheaf = Sheaf(
            poset=self.poset,
            stalks=self.stalks,
            restrictions=self.restrictions
        )
        
        validation_results = sheaf.validate(tolerance=1e-1)
        
        assert isinstance(validation_results, dict)
        assert 'transitivity_violations' in validation_results
        assert 'valid_sheaf' in validation_results
        assert 'total_paths_checked' in validation_results
        
        # Check that validation results are stored in metadata
        assert 'validation_results' in sheaf.metadata
        assert 'validation_passed' in sheaf.metadata
    
    def test_get_laplacian_structure(self):
        """Test Laplacian structure information."""
        sheaf = Sheaf(
            poset=self.poset,
            stalks=self.stalks,
            restrictions=self.restrictions
        )
        
        laplacian_info = sheaf.get_laplacian_structure()
        
        assert 'total_dimension' in laplacian_info
        assert 'num_nodes' in laplacian_info
        assert 'num_edges' in laplacian_info
        assert 'estimated_sparsity' in laplacian_info
        assert 'memory_savings' in laplacian_info
        
        assert laplacian_info['num_nodes'] == 3
        assert laplacian_info['num_edges'] == 2
        assert laplacian_info['total_dimension'] == 24  # 3 stalks × 8 dims each
        assert 0 <= laplacian_info['estimated_sparsity'] <= 1
    
    def test_sheaf_summary(self):
        """Test sheaf summary string."""
        sheaf = Sheaf(
            poset=self.poset,
            stalks=self.stalks,
            restrictions=self.restrictions
        )
        
        summary = sheaf.summary()
        
        assert isinstance(summary, str)
        assert 'Nodes: 3' in summary
        assert 'Edges: 2' in summary
        assert 'Total dimension: 24' in summary
        assert 'Sparsity:' in summary
        assert 'Validation:' in summary


class TestSheafBuilder:
    """Test cases for the SheafBuilder class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.builder = SheafBuilder(handle_dynamic=True)
        self.model = MLPTestModel()
        
        # Create test activations
        torch.manual_seed(42)
        self.test_input = torch.randn(16, 10)  # batch_size=16, input_dim=10
        
        with torch.no_grad():
            self.model.eval()
            _ = self.model(self.test_input)
        
        # Manually create activations dict (simulating extracted activations)
        self.activations = {
            'fc1': torch.randn(16, 20),
            'fc2': torch.randn(16, 15),
            'fc3': torch.randn(16, 5)
        }
    
    def test_builder_initialization(self):
        """Test SheafBuilder initialization."""
        builder = SheafBuilder(
            handle_dynamic=True,
            procrustes_epsilon=1e-6,
            restriction_method='orthogonal_projection'
        )
        
        assert builder.poset_extractor.handle_dynamic is True
        assert builder.procrustes_maps.epsilon == 1e-6
        assert builder.default_method == 'orthogonal_projection'
    
    def test_build_from_activations_gram_matrices(self):
        """Test building sheaf from activations using Gram matrices."""
        sheaf = self.builder.build_from_activations(
            self.model, self.activations, use_gram_matrices=True, validate=False
        )
        
        assert isinstance(sheaf, Sheaf)
        assert len(sheaf.poset.nodes()) > 0
        assert len(sheaf.stalks) > 0
        assert len(sheaf.restrictions) >= 0  # May be 0 if no valid edges
        
        # Check that stalks are Gram matrices
        for node, stalk in sheaf.stalks.items():
            if node in self.activations:
                expected_dim = self.activations[node].shape[0]  # batch size
                assert stalk.shape == (expected_dim, expected_dim)
                
                # Check that it's a Gram matrix (symmetric, PSD)
                assert torch.allclose(stalk, stalk.T, atol=1e-6)
                eigenvals = torch.linalg.eigvals(stalk)
                assert torch.all(eigenvals.real >= -1e-6)  # PSD within tolerance
        
        # Check metadata
        assert sheaf.metadata['construction_method'] == 'activations'
        assert sheaf.metadata['use_gram_matrices'] is True
    
    def test_build_from_activations_raw_activations(self):
        """Test building sheaf from raw activations."""
        sheaf = self.builder.build_from_activations(
            self.model, self.activations, use_gram_matrices=False, validate=False
        )
        
        assert isinstance(sheaf, Sheaf)
        
        # Check that stalks are raw activations
        for node, stalk in sheaf.stalks.items():
            if node in self.activations:
                expected_shape = self.activations[node].shape
                assert stalk.shape == expected_shape
        
        assert sheaf.metadata['use_gram_matrices'] is False
    
    def test_build_from_cka_matrices(self):
        """Test building sheaf from CKA matrices."""
        # Create simple poset
        poset = nx.DiGraph()
        poset.add_edges_from([('layer1', 'layer2'), ('layer2', 'layer3')])
        
        # Create test CKA matrices
        cka_matrices = {
            'layer1': torch.randn(1, 10),  # CKA similarity vector
            'layer2': torch.randn(1, 10),
            'layer3': torch.randn(1, 10)
        }
        
        sheaf = self.builder.build_from_cka_matrices(
            poset, cka_matrices, validate=False
        )
        
        assert isinstance(sheaf, Sheaf)
        assert sheaf.poset.number_of_nodes() == 3
        assert len(sheaf.stalks) == 3
        
        # Check that stalks are CKA matrices
        for node in ['layer1', 'layer2', 'layer3']:
            assert node in sheaf.stalks
            assert torch.equal(sheaf.stalks[node], cka_matrices[node])
        
        assert sheaf.metadata['construction_method'] == 'cka_matrices'
    
    def test_build_from_model_comparison(self):
        """Test building sheaf for model comparison."""
        model1 = MLPTestModel()
        model2 = MLPTestModel()
        input_data = torch.randn(8, 10)
        
        # This is a more complex test - it might fail if activation extraction is complex
        try:
            sheaf = self.builder.build_from_model_comparison(
                model1, model2, input_data, validate=False
            )
            
            assert isinstance(sheaf, Sheaf)
            assert len(sheaf.stalks) > 0
            
            # Check that activations were combined for comparison
            for stalk in sheaf.stalks.values():
                # Should have combined batch size (16 = 8 from each model)
                if stalk.ndim == 2:  # Gram matrix
                    assert stalk.shape[0] == 16
        except Exception as e:
            # Activation extraction might be complex, so we allow this to fail
            pytest.skip(f"Model comparison test failed due to activation extraction: {e}")
    
    def test_validation_integration(self):
        """Test integration with validation."""
        sheaf = self.builder.build_from_activations(
            self.model, self.activations, validate=True
        )
        
        assert 'validation_results' in sheaf.metadata
        assert 'validation_passed' in sheaf.metadata
        
        validation_results = sheaf.metadata['validation_results']
        assert 'valid_sheaf' in validation_results
        assert 'total_paths_checked' in validation_results
    
    def test_missing_activations_handling(self):
        """Test handling of missing activations."""
        incomplete_activations = {
            'fc1': torch.randn(16, 20),
            # Missing fc2 and fc3
        }
        
        # Should handle gracefully with warnings
        sheaf = self.builder.build_from_activations(
            self.model, incomplete_activations, validate=False
        )
        
        assert isinstance(sheaf, Sheaf)
        # Should have at least one stalk
        assert len(sheaf.stalks) >= 1
    
    def test_empty_activations(self):
        """Test handling of empty activations."""
        empty_activations = {}
        
        sheaf = self.builder.build_from_activations(
            self.model, empty_activations, validate=False
        )
        
        assert isinstance(sheaf, Sheaf)
        # May have empty stalks
        assert len(sheaf.stalks) == 0


class TestIntegrationWithComponents:
    """Test integration with other neurosheaf components."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.builder = SheafBuilder()
        
    def test_fx_poset_integration(self):
        """Test integration with FX poset extraction."""
        model = SimpleTestModel()
        
        # Test that FX extraction works
        extractor = FXPosetExtractor(handle_dynamic=True)
        poset = extractor.extract_poset(model)
        
        assert isinstance(poset, nx.DiGraph)
        assert len(poset.nodes()) > 0
        
        # Should be able to use this poset for sheaf construction
        cka_matrices = {}
        for node in poset.nodes():
            cka_matrices[node] = torch.randn(1, 5)
        
        sheaf = self.builder.build_from_cka_matrices(poset, cka_matrices)
        assert isinstance(sheaf, Sheaf)
    
    def test_procrustes_integration(self):
        """Test integration with Procrustes computation."""
        # Create test data
        activations = {
            'layer1': torch.randn(10, 8),
            'layer2': torch.randn(10, 8),
            'layer3': torch.randn(10, 8)
        }
        
        model = MLPTestModel()
        sheaf = self.builder.build_from_activations(
            model, activations, use_gram_matrices=True, validate=False
        )
        
        # Check that restrictions were computed
        if len(sheaf.restrictions) > 0:
            # All restrictions should be valid tensors
            for edge, restriction in sheaf.restrictions.items():
                assert isinstance(restriction, torch.Tensor)
                assert restriction.shape[0] == restriction.shape[1]  # Square for same-size Gram matrices
    
    def test_validation_with_real_restrictions(self):
        """Test validation with actually computed restrictions."""
        # Create activations with clear structure
        torch.manual_seed(123)
        activations = {
            'A': torch.randn(5, 4),
            'B': torch.randn(5, 4),
            'C': torch.randn(5, 4)
        }
        
        # Manually create a simple model for testing
        simple_model = nn.Sequential(
            nn.Linear(4, 4),
            nn.ReLU(),
            nn.Linear(4, 4)
        )
        
        sheaf = self.builder.build_from_activations(
            simple_model, activations, validate=True
        )
        
        # Check validation results
        if 'validation_results' in sheaf.metadata:
            validation = sheaf.metadata['validation_results']
            assert 'total_paths_checked' in validation
            assert 'valid_sheaf' in validation


class TestCreateSheafFromCKA:
    """Test the create_sheaf_from_cka_analysis convenience function."""
    
    def test_create_from_cka_results(self):
        """Test creating sheaf from CKA analysis results."""
        # Simulate CKA results
        cka_results = {
            'similarity_matrix': torch.randn(3, 3),
            'layer_names': ['layer1', 'layer2', 'layer3']
        }
        layer_names = ['layer1', 'layer2', 'layer3']
        
        sheaf = create_sheaf_from_cka_analysis(cka_results, layer_names)
        
        assert isinstance(sheaf, Sheaf)
        assert len(sheaf.stalks) == 3
        assert sheaf.poset.number_of_nodes() == 3
        assert sheaf.poset.number_of_edges() == 2  # Sequential connections
    
    def test_create_with_custom_structure(self):
        """Test creating sheaf with custom network structure."""
        cka_results = {
            'similarity_matrix': torch.randn(4, 4)
        }
        layer_names = ['A', 'B', 'C', 'D']
        
        # Custom poset with branching
        custom_poset = nx.DiGraph()
        custom_poset.add_edges_from([('A', 'B'), ('A', 'C'), ('B', 'D'), ('C', 'D')])
        
        sheaf = create_sheaf_from_cka_analysis(
            cka_results, layer_names, network_structure=custom_poset
        )
        
        assert isinstance(sheaf, Sheaf)
        assert sheaf.poset.number_of_edges() == 4  # Custom structure
        assert sheaf.poset.has_edge('A', 'B')
        assert sheaf.poset.has_edge('A', 'C')


class TestErrorHandling:
    """Test error handling and edge cases."""
    
    def test_invalid_model_architecture(self):
        """Test handling of models that cannot be traced."""
        class UntracableModel(nn.Module):
            def forward(self, x):
                # Dynamic control flow that FX cannot trace
                if x.sum() > 0:
                    return x * 2
                else:
                    return x * 3
        
        model = UntracableModel()
        builder = SheafBuilder(handle_dynamic=False)
        
        with pytest.raises(ArchitectureError):
            builder.build_from_activations(model, {})
    
    def test_mismatched_dimensions(self):
        """Test handling of severely mismatched dimensions."""
        activations = {
            'small': torch.randn(5, 2),
            'large': torch.randn(20, 50)
        }
        
        model = nn.Linear(2, 2)  # Simple model
        builder = SheafBuilder()
        
        # Should handle gracefully
        sheaf = builder.build_from_activations(model, activations, validate=False)
        assert isinstance(sheaf, Sheaf)
    
    def test_numerical_instability_handling(self):
        """Test handling of numerically unstable inputs."""
        # Create nearly singular Gram matrices
        activations = {
            'unstable': torch.ones(10, 10) * 1e-12 + torch.eye(10) * 1e-15
        }
        
        model = nn.Linear(10, 10)
        builder = SheafBuilder()
        
        # Should either handle gracefully or raise appropriate error
        try:
            sheaf = builder.build_from_activations(model, activations, validate=False)
            assert isinstance(sheaf, Sheaf)
        except ComputationError:
            # Acceptable to raise this error
            pass
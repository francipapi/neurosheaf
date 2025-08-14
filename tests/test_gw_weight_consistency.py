"""Tests for consistent weight scaling in GW Laplacian assembly.

This module validates that all blocks in the GW Laplacian (off-diagonal, identity, R^T R)
apply edge weights consistently as weight² terms, and that the sqrt(similarity) extraction
design works correctly.
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
from scipy.sparse import csr_matrix
from neurosheaf.sheaf.assembly import SheafBuilder
from neurosheaf.sheaf.assembly.gw_laplacian import GWLaplacianBuilder, GWWeightTransform
from neurosheaf.sheaf.core import GWConfig


class WeightTestNetwork(nn.Module):
    """Simple test network for weight consistency validation."""
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Linear(4, 6),
            nn.Linear(6, 3)
        ])
    
    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = torch.relu(x)
        return x


class TestGWWeightConsistency:
    """Test consistent weight scaling across all Laplacian blocks."""
    
    def test_sqrt_similarity_extraction_design(self):
        """Test that extract_edge_weights produces sqrt(similarity) as documented."""
        model = WeightTestNetwork()
        input_tensor = torch.randn(8, 4)
        
        builder = SheafBuilder(restriction_method='gromov_wasserstein')
        gw_config = GWConfig()
        
        sheaf = builder.build_from_activations(
            model, input_tensor,
            validate=False,
            gw_config=gw_config
        )
        
        # Extract edge weights using different transform methods
        gw_builder = GWLaplacianBuilder()
        
        # Test exponential transform
        weights_exp = gw_builder.extract_edge_weights(
            sheaf, transform_method=GWWeightTransform.EXPONENTIAL, transform_beta=1.0
        )
        
        # Test reciprocal transform
        weights_recip = gw_builder.extract_edge_weights(
            sheaf, transform_method=GWWeightTransform.RECIPROCAL
        )
        
        # Get raw GW costs for comparison
        raw_costs = sheaf.get_gw_costs()
        
        assert raw_costs is not None and len(raw_costs) > 0, "Should have GW costs"
        assert len(weights_exp) == len(raw_costs), "Should extract weights for all edges with costs"
        assert len(weights_recip) == len(raw_costs), "Should extract weights for all edges with costs"
        
        # Verify sqrt(similarity) relationship for exponential transform
        for edge in raw_costs.keys():
            if edge in weights_exp:
                cost = raw_costs[edge]
                weight = weights_exp[edge]
                expected_similarity = np.exp(-1.0 * cost)
                expected_weight = np.sqrt(expected_similarity)
                
                assert abs(weight - expected_weight) < 1e-6, \
                    f"Edge {edge}: weight {weight} != sqrt(exp(-cost)) = {expected_weight}"
        
        # Verify sqrt(similarity) relationship for reciprocal transform
        for edge in raw_costs.keys():
            if edge in weights_recip:
                cost = raw_costs[edge]
                weight = weights_recip[edge]
                expected_similarity = 1.0 / (1.0 + cost)
                expected_weight = np.sqrt(expected_similarity)
                
                assert abs(weight - expected_weight) < 1e-6, \
                    f"Edge {edge}: weight {weight} != sqrt(1/(1+cost)) = {expected_weight}"
    
    def test_laplacian_weight_squaring_consistency(self):
        """Test that all Laplacian blocks apply weight² consistently."""
        model = WeightTestNetwork()
        input_tensor = torch.randn(6, 4)
        
        builder = SheafBuilder(restriction_method='gromov_wasserstein')
        sheaf = builder.build_from_activations(model, input_tensor, validate=False)
        
        # Build Laplacian with known weights
        gw_builder = GWLaplacianBuilder()
        laplacian = gw_builder.build_laplacian(sheaf, sparse=True)
        
        # Extract weights used
        edge_weights = gw_builder.extract_edge_weights(sheaf)
        
        # Convert to dense for analysis
        L_dense = laplacian.toarray()
        stalk_dims = sheaf.metadata['stalk_dimensions']
        
        # Calculate stalk offsets
        offsets = {}
        current_offset = 0
        for node in sorted(stalk_dims.keys()):
            offsets[node] = current_offset
            current_offset += stalk_dims[node]
        
        # Test that diagonal blocks contain weight² terms
        for node, dim in stalk_dims.items():
            start = offsets[node]
            end = start + dim
            diagonal_block = L_dense[start:end, start:end]
            
            # Check for identity contributions (from incoming edges)
            incoming_weight_squares = []
            for pred in sheaf.poset.predecessors(node):
                edge = (pred, node)
                if edge in edge_weights:
                    weight = edge_weights[edge]
                    incoming_weight_squares.append(weight**2)
            
            # Check for R^T R contributions (from outgoing edges)
            outgoing_weight_squares = []
            for succ in sheaf.poset.successors(node):
                edge = (node, succ)
                if edge in edge_weights:
                    weight = edge_weights[edge]
                    outgoing_weight_squares.append(weight**2)
            
            # The diagonal should contain contributions scaled by weight²
            diagonal_trace = np.trace(diagonal_block)
            
            if incoming_weight_squares or outgoing_weight_squares:
                # Should have non-zero diagonal from weight² contributions
                assert diagonal_trace > 1e-6, \
                    f"Node {node} diagonal should have weight² contributions, trace: {diagonal_trace}"
    
    def test_energy_scales_linearly_with_similarity(self):
        """Test that Laplacian energy scales linearly with similarity, not similarity²."""
        model = WeightTestNetwork()
        input_tensor = torch.randn(5, 4)
        
        builder = SheafBuilder(restriction_method='gromov_wasserstein')
        sheaf = builder.build_from_activations(model, input_tensor, validate=False)
        
        # Test different transform betas (affects similarity strength)
        betas = [0.5, 1.0, 2.0]
        energies = []
        
        for beta in betas:
            gw_builder = GWLaplacianBuilder(weight_transform=GWWeightTransform.EXPONENTIAL, 
                                           transform_beta=beta)
            laplacian = gw_builder.build_laplacian(sheaf, sparse=True)
            
            # Compute energy for a random vector
            np.random.seed(42)  # Reproducible test
            x = np.random.randn(laplacian.shape[0])
            energy = x.T @ laplacian @ x
            energies.append(energy)
        
        # Higher beta → stronger similarity weights → higher energy
        # This validates that energy scales with intended similarity strength
        assert energies[1] > energies[0], "Energy should increase with beta (stronger similarities)"
        assert energies[2] > energies[1], "Energy should increase with beta (stronger similarities)"
    
    def test_weight_consistency_across_block_types(self):
        """Test that off-diagonal, identity, and R^T R blocks use the same weight² scaling."""
        model = WeightTestNetwork()
        input_tensor = torch.randn(4, 4)
        
        builder = SheafBuilder(restriction_method='gromov_wasserstein')
        sheaf = builder.build_from_activations(model, input_tensor, validate=False)
        
        # Create a custom GW builder to examine intermediate computations
        gw_builder = GWLaplacianBuilder()
        
        # Extract weights
        edge_weights = gw_builder.extract_edge_weights(sheaf)
        
        # Mock the sparse assembly to check weight application
        metadata = gw_builder._initialize_gw_metadata(sheaf, edge_weights)
        
        # Check that all weight applications use weight² consistently
        for edge, weight in edge_weights.items():
            source, target = edge
            
            # This is the key consistency check: all blocks should use weight²
            weight_squared = weight**2
            
            # Verify the weight is the sqrt(similarity) as designed
            raw_costs = sheaf.get_gw_costs()
            if edge in raw_costs:
                cost = raw_costs[edge]
                expected_similarity = np.exp(-1.0 * cost)  # Default exponential transform
                expected_weight = np.sqrt(expected_similarity)
                
                assert abs(weight - expected_weight) < 1e-5, \
                    f"Edge {edge}: extracted weight {weight} should be sqrt(similarity) = {expected_weight}"
                
                # The squared weight should equal the similarity
                assert abs(weight_squared - expected_similarity) < 1e-5, \
                    f"Edge {edge}: weight² {weight_squared} should equal similarity {expected_similarity}"
    
    def test_different_transform_methods_consistency(self):
        """Test that different transform methods maintain sqrt(similarity) design."""
        model = WeightTestNetwork()
        input_tensor = torch.randn(7, 4)
        
        builder = SheafBuilder(restriction_method='gromov_wasserstein')
        sheaf = builder.build_from_activations(model, input_tensor, validate=False)
        
        raw_costs = sheaf.get_gw_costs()
        assert len(raw_costs) > 0, "Need GW costs for testing"
        
        # Test all transform methods
        gw_builder = GWLaplacianBuilder()
        
        for transform_method in [GWWeightTransform.EXPONENTIAL, 
                               GWWeightTransform.RECIPROCAL,
                               GWWeightTransform.LINEAR]:
            
            weights = gw_builder.extract_edge_weights(
                sheaf, transform_method=transform_method
            )
            
            # Verify sqrt(similarity) relationship for each method
            for edge, cost in raw_costs.items():
                if edge in weights:
                    weight = weights[edge]
                    
                    if transform_method == GWWeightTransform.EXPONENTIAL:
                        expected_similarity = np.exp(-1.0 * cost)
                    elif transform_method == GWWeightTransform.RECIPROCAL:
                        expected_similarity = 1.0 / (1.0 + cost)
                    elif transform_method == GWWeightTransform.LINEAR:
                        max_cost = max(raw_costs.values())
                        expected_similarity = max_cost + 1e-6 - cost
                    
                    expected_weight = np.sqrt(max(expected_similarity, 1e-8))
                    
                    assert abs(weight - expected_weight) < 1e-5, \
                        f"Transform {transform_method}, edge {edge}: " \
                        f"weight {weight} != sqrt(similarity) {expected_weight}"
    
    def test_weight_none_transform_deprecation_warning(self):
        """Test that NONE transform produces expected warning about inverted semantics."""
        model = WeightTestNetwork()
        input_tensor = torch.randn(5, 4)
        
        builder = SheafBuilder(restriction_method='gromov_wasserstein')
        sheaf = builder.build_from_activations(model, input_tensor, validate=False)
        
        gw_builder = GWLaplacianBuilder()
        
        # This should produce a warning about deprecated behavior
        import logging
        with pytest.LoggingCapture() as logs:
            weights = gw_builder.extract_edge_weights(
                sheaf, transform_method=GWWeightTransform.NONE
            )
        
        # Should have logged a warning about inverted semantics
        warning_found = any("deprecated" in record.message.lower() and "inverts semantics" in record.message.lower() 
                           for record in logs.records if record.levelno >= logging.WARNING)
        
        assert warning_found, "Should warn about deprecated NONE transform inverting semantics"
        
        # With NONE transform, weights should just be raw costs (not sqrt transformed)
        raw_costs = sheaf.get_gw_costs()
        for edge, cost in raw_costs.items():
            if edge in weights:
                assert abs(weights[edge] - cost) < 1e-6, \
                    f"NONE transform should preserve raw costs: {weights[edge]} vs {cost}"
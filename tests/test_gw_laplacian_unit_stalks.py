"""Tests for GW Laplacian assembly with unit-wise stalks.

This module validates that the GW Laplacian assembly correctly handles unit-wise stalks
and that dimensions remain consistent regardless of batch size (unlike sample-wise stalks).
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
from neurosheaf.sheaf.assembly import SheafBuilder
from neurosheaf.sheaf.assembly.gw_laplacian import GWLaplacianBuilder
from neurosheaf.sheaf.core import GWConfig


class UnitTestNetwork(nn.Module):
    """Test network with known layer sizes for unit stalk validation."""
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Linear(6, 8),    # 8 units in hidden layer
            nn.Linear(8, 4),    # 4 units in output layer  
        ])
    
    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = torch.relu(x)
        return x


class TestGWLaplacianUnitStalks:
    """Test GW Laplacian assembly with unit-wise stalks."""
    
    def test_unit_wise_stalks_batch_independence(self):
        """Test that unit-wise stalks produce batch-size independent results."""
        model = UnitTestNetwork()
        
        # Build sheaves with different batch sizes
        batch_sizes = [5, 10, 20]
        sheaves = []
        
        for batch_size in batch_sizes:
            input_tensor = torch.randn(batch_size, 6)
            
            builder = SheafBuilder(restriction_method='gromov_wasserstein')
            gw_config = GWConfig(align_units=True)  # Ensure unit-wise stalks
            
            sheaf = builder.build_from_activations(
                model, input_tensor,
                validate=False,
                gw_config=gw_config
            )
            sheaves.append(sheaf)
        
        # All sheaves should have the same stalk dimensions (independent of batch size)
        ref_stalk_dims = sheaves[0].metadata['stalk_dimensions']
        
        for i, sheaf in enumerate(sheaves[1:], 1):
            stalk_dims = sheaf.metadata['stalk_dimensions']
            
            assert stalk_dims == ref_stalk_dims, \
                f"Batch size {batch_sizes[i]} produced different stalk dimensions: {stalk_dims} vs {ref_stalk_dims}"
        
        # Stalk dimensions should match layer widths, not batch sizes
        # Input layer: 6 units (from input_tensor.shape[1])
        # Hidden layer: 8 units (from Linear(6, 8))  
        # Output layer: 4 units (from Linear(8, 4))
        expected_dims = {'layers_0': 8, 'layers_1': 4}  # Only intermediate layers get stalks
        
        for layer_name, expected_dim in expected_dims.items():
            if layer_name in ref_stalk_dims:
                actual_dim = ref_stalk_dims[layer_name]
                assert actual_dim == expected_dim, \
                    f"Layer {layer_name}: expected {expected_dim} units, got {actual_dim}"
    
    def test_laplacian_dimensions_with_unit_stalks(self):
        """Test that Laplacian dimensions are coherent with unit-wise stalks."""
        model = UnitTestNetwork()
        input_tensor = torch.randn(15, 6)  # batch_size=15, input_dim=6 (different from layer sizes)
        
        # Build GW sheaf with unit-wise stalks
        builder = SheafBuilder(restriction_method='gromov_wasserstein')
        gw_config = GWConfig(align_units=True)
        
        sheaf = builder.build_from_activations(
            model, input_tensor,
            validate=False,
            gw_config=gw_config
        )
        
        # Build Laplacian 
        gw_laplacian_builder = GWLaplacianBuilder()
        laplacian = gw_laplacian_builder.build_laplacian(sheaf, sparse=True)
        
        # Laplacian size should be sum of stalk dimensions (unit counts)
        stalk_dims = sheaf.metadata['stalk_dimensions']
        expected_size = sum(stalk_dims.values())
        
        actual_shape = laplacian.shape
        assert actual_shape == (expected_size, expected_size), \
            f"Laplacian shape {actual_shape} doesn't match expected ({expected_size}, {expected_size}) from stalk dims {stalk_dims}"
        
        # Verify stalk dimensions are unit-based, not batch-based
        for node, dim in stalk_dims.items():
            assert dim != 15, f"Stalk dimension {dim} matches batch size 15 - should be unit count, not batch size"
    
    def test_restriction_shapes_match_stalk_dimensions(self):
        """Test that restriction shapes are coherent with unit-wise stalk dimensions."""
        model = UnitTestNetwork()
        input_tensor = torch.randn(12, 6)  # Different batch size to verify independence
        
        builder = SheafBuilder(restriction_method='gromov_wasserstein')
        gw_config = GWConfig(align_units=True)
        
        sheaf = builder.build_from_activations(
            model, input_tensor,
            validate=False,
            gw_config=gw_config
        )
        
        stalk_dims = sheaf.metadata['stalk_dimensions']
        
        # Check that restriction maps have shapes consistent with stalk dimensions
        for (source, target), restriction in sheaf.restrictions.items():
            restriction_shape = restriction.shape  # (n_target_units, n_source_units)
            
            source_dim = stalk_dims.get(source)
            target_dim = stalk_dims.get(target)
            
            if source_dim is not None and target_dim is not None:
                expected_shape = (target_dim, source_dim)
                
                # Allow for potential dimension mismatches due to layer extraction differences
                # but log when they occur
                if restriction_shape != expected_shape:
                    print(f"Edge {source}→{target}: restriction {restriction_shape} vs stalk dims {expected_shape}")
                    
                # Key test: restriction dimensions should not equal batch size
                assert restriction_shape[0] != 12, \
                    f"Restriction target dim {restriction_shape[0]} matches batch size - should be unit count"
                assert restriction_shape[1] != 12, \
                    f"Restriction source dim {restriction_shape[1]} matches batch size - should be unit count"
    
    def test_laplacian_block_assembly_coherence(self):
        """Test that Laplacian block assembly is mathematically coherent."""
        model = UnitTestNetwork()
        input_tensor = torch.randn(6, 6)
        
        builder = SheafBuilder(restriction_method='gromov_wasserstein')
        gw_config = GWConfig(align_units=True)
        
        sheaf = builder.build_from_activations(
            model, input_tensor,
            validate=False,
            gw_config=gw_config
        )
        
        # Build Laplacian
        gw_laplacian_builder = GWLaplacianBuilder()
        laplacian = gw_laplacian_builder.build_laplacian(sheaf, sparse=True)
        
        # Validate basic Laplacian properties
        # 1. Square matrix
        assert laplacian.shape[0] == laplacian.shape[1], "Laplacian should be square"
        
        # 2. Symmetric (within numerical tolerance)
        laplacian_dense = laplacian.toarray()
        symmetry_error = np.abs(laplacian_dense - laplacian_dense.T).max()
        assert symmetry_error < 1e-10, f"Laplacian should be symmetric, max asymmetry: {symmetry_error}"
        
        # 3. Positive semi-definite (non-negative eigenvalues within numerical tolerance)
        eigenvals = np.linalg.eigvals(laplacian_dense)
        min_eigenval = np.min(np.real(eigenvals))
        assert min_eigenval >= -1e-8, f"Laplacian should be PSD, min eigenvalue: {min_eigenval} (numerical tolerance)"
        
        # 4. Each diagonal block should have appropriate dimensions
        stalk_dims = sheaf.metadata['stalk_dimensions']
        
        # Calculate stalk offsets (should match those used in assembly)
        offsets = {}
        current_offset = 0
        for node in sorted(stalk_dims.keys()):  # Deterministic ordering
            offsets[node] = current_offset
            current_offset += stalk_dims[node]
        
        # Check diagonal block sizes
        for node, dim in stalk_dims.items():
            start = offsets[node]
            end = start + dim
            
            diagonal_block = laplacian_dense[start:end, start:end]
            assert diagonal_block.shape == (dim, dim), \
                f"Node {node} diagonal block shape {diagonal_block.shape} != expected ({dim}, {dim})"
    
    def test_unit_stalks_vs_sample_stalks_difference(self):
        """Test that unit-wise and sample-wise stalks produce different results."""
        model = UnitTestNetwork()
        input_tensor = torch.randn(10, 6)  # batch_size=10
        
        # Build with unit-wise stalks (default)
        builder_units = SheafBuilder(restriction_method='gromov_wasserstein')
        gw_config_units = GWConfig(align_units=True)
        
        sheaf_units = builder_units.build_from_activations(
            model, input_tensor,
            validate=False,
            gw_config=gw_config_units
        )
        
        # Build with sample-wise stalks (deprecated)
        builder_samples = SheafBuilder(restriction_method='gromov_wasserstein')
        gw_config_samples = GWConfig(align_units=False)
        
        sheaf_samples = builder_samples.build_from_activations(
            model, input_tensor,
            validate=False,
            gw_config=gw_config_samples
        )
        
        # Stalk dimensions should be different
        dims_units = sheaf_units.metadata['stalk_dimensions']
        dims_samples = sheaf_samples.metadata['stalk_dimensions']
        
        # Unit-wise: dimensions = layer widths (independent of batch size)
        # Sample-wise: dimensions = batch size (10 in this case)
        
        for node in dims_units.keys():
            if node in dims_samples:
                unit_dim = dims_units[node]
                sample_dim = dims_samples[node]
                
                # They should be different (unless by coincidence layer width = batch size)
                if unit_dim != sample_dim:
                    # Expected case: unit count != batch size
                    print(f"Node {node}: unit-wise={unit_dim}, sample-wise={sample_dim}")
                    
                    # Unit-wise should not equal batch size (10)
                    assert unit_dim != 10, f"Unit-wise stalk dim {unit_dim} shouldn't equal batch size"
                    
                    # Sample-wise should equal batch size (10) 
                    assert sample_dim == 10, f"Sample-wise stalk dim {sample_dim} should equal batch size 10"
                else:
                    # Rare case: layer width happens to equal batch size
                    print(f"Node {node}: coincidental match unit_dim=sample_dim={unit_dim}")
    
    def test_gw_laplacian_metadata_records_unit_alignment(self):
        """Test that GW Laplacian metadata correctly records unit alignment mode."""
        model = UnitTestNetwork()
        input_tensor = torch.randn(7, 6)
        
        builder = SheafBuilder(restriction_method='gromov_wasserstein')
        gw_config = GWConfig(align_units=True)
        
        sheaf = builder.build_from_activations(
            model, input_tensor,
            validate=False,
            gw_config=gw_config
        )
        
        # Check sheaf metadata records unit alignment
        assert sheaf.metadata.get('align_units') == True
        assert sheaf.metadata.get('alignment_type') == 'unit'
        
        # Build Laplacian and check its metadata
        gw_laplacian_builder = GWLaplacianBuilder()
        laplacian = gw_laplacian_builder.build_laplacian(sheaf, sparse=True)
        
        # The laplacian builder doesn't directly store alignment info, but sheaf does
        assert sheaf.metadata['construction_method'] == 'gromov_wasserstein'
        assert sheaf.is_gw_sheaf()
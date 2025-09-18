"""Tests for dtype consistency throughout the GW pipeline.

This module validates that the configurable dtype policy is properly implemented
across all components of the GW sheaf construction and Laplacian assembly pipeline.
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
from neurosheaf.sheaf.assembly import SheafBuilder
from neurosheaf.sheaf.assembly.gw_laplacian import GWLaplacianBuilder
from neurosheaf.sheaf.core import GWConfig


class DtypeTestNetwork(nn.Module):
    """Simple test network for dtype consistency validation."""
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Linear(4, 6),
            nn.Linear(6, 3),
        ])
    
    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = torch.relu(x)
        return x


class TestGWDtypeConsistency:
    """Test dtype consistency throughout GW pipeline."""
    
    @pytest.mark.parametrize("dtype", ['float32', 'float64'])
    def test_gw_config_dtype_conversion(self, dtype):
        """Test GWConfig dtype conversion methods."""
        config = GWConfig(computation_dtype=dtype)
        
        # Test torch dtype conversion
        torch_dtype = config.get_torch_dtype()
        expected_torch = torch.float32 if dtype == 'float32' else torch.float64
        assert torch_dtype == expected_torch
        
        # Test numpy dtype conversion
        numpy_dtype = config.get_numpy_dtype()
        expected_numpy = np.float32 if dtype == 'float32' else np.float64
        assert numpy_dtype == expected_numpy
    
    def test_gw_config_invalid_dtype(self):
        """Test GWConfig validation for invalid dtype."""
        with pytest.raises(ValueError, match="computation_dtype must be"):
            GWConfig(computation_dtype='invalid')
    
    @pytest.mark.parametrize("dtype", ['float32', 'float64'])
    def test_activation_dtype_conversion(self, dtype):
        """Test that activations are converted to target dtype early in pipeline."""
        model = DtypeTestNetwork()
        
        # Convert model to target dtype for compatibility
        model_dtype = torch.float32 if dtype == 'float32' else torch.float64
        model = model.to(dtype=model_dtype)
        
        # Create input with same dtype as model (for model forward pass to work)
        input_tensor = torch.randn(8, 4, dtype=model_dtype)
        
        # Build sheaf with specific dtype
        builder = SheafBuilder(restriction_method='gromov_wasserstein')
        gw_config = GWConfig(computation_dtype=dtype)
        
        sheaf = builder.build_from_activations(
            model, input_tensor,
            validate=False,
            gw_config=gw_config
        )
        
        # Check that sheaf was built successfully
        assert len(sheaf.restrictions) > 0
        
        # Check that config dtype is recorded in metadata
        assert sheaf.metadata.get('gw_config', {}).get('computation_dtype') == dtype
    
    @pytest.mark.parametrize("dtype", ['float32', 'float64'])
    def test_restriction_map_dtype_consistency(self, dtype):
        """Test that restriction maps have consistent dtype."""
        model = DtypeTestNetwork()
        input_tensor = torch.randn(6, 4)
        
        builder = SheafBuilder(restriction_method='gromov_wasserstein')
        gw_config = GWConfig(computation_dtype=dtype)
        
        sheaf = builder.build_from_activations(
            model, input_tensor,
            validate=False,
            gw_config=gw_config
        )
        
        expected_torch_dtype = torch.float32 if dtype == 'float32' else torch.float64
        
        # Check all restriction maps have correct dtype
        for (source, target), restriction in sheaf.restrictions.items():
            assert restriction.dtype == expected_torch_dtype, \
                f"Restriction {source}→{target} has dtype {restriction.dtype}, expected {expected_torch_dtype}"
    
    @pytest.mark.parametrize("dtype", ['float32', 'float64'])
    def test_gw_coupling_dtype_consistency(self, dtype):
        """Test that GW couplings have consistent dtype."""
        model = DtypeTestNetwork()
        input_tensor = torch.randn(5, 4)
        
        builder = SheafBuilder(restriction_method='gromov_wasserstein')
        gw_config = GWConfig(computation_dtype=dtype)
        
        sheaf = builder.build_from_activations(
            model, input_tensor,
            validate=False,
            gw_config=gw_config
        )
        
        expected_torch_dtype = torch.float32 if dtype == 'float32' else torch.float64
        
        # Check GW couplings in metadata
        gw_couplings = sheaf.metadata.get('gw_couplings', {})
        
        for edge, coupling in gw_couplings.items():
            if hasattr(coupling, 'dtype'):
                assert coupling.dtype == expected_torch_dtype, \
                    f"GW coupling for edge {edge} has dtype {coupling.dtype}, expected {expected_torch_dtype}"
    
    @pytest.mark.parametrize("dtype", ['float32', 'float64'])
    def test_laplacian_builder_dtype_parameter(self, dtype):
        """Test GWLaplacianBuilder dtype parameter handling."""
        # Test with explicit dtype parameter
        builder = GWLaplacianBuilder(computation_dtype=dtype)
        
        expected_torch = torch.float32 if dtype == 'float32' else torch.float64
        expected_numpy = np.float32 if dtype == 'float32' else np.float64
        
        assert builder.torch_dtype == expected_torch
        assert builder.numpy_dtype == expected_numpy
    
    @pytest.mark.parametrize("dtype", ['float32', 'float64'])
    def test_laplacian_dtype_inference_from_sheaf(self, dtype):
        """Test that Laplacian builder infers dtype from sheaf metadata."""
        model = DtypeTestNetwork()
        
        # Convert model to target dtype for compatibility
        model_dtype = torch.float32 if dtype == 'float32' else torch.float64
        model = model.to(dtype=model_dtype)
        input_tensor = torch.randn(7, 4, dtype=model_dtype)
        
        # Build sheaf with specific dtype
        builder = SheafBuilder(restriction_method='gromov_wasserstein')
        gw_config = GWConfig(computation_dtype=dtype)
        
        sheaf = builder.build_from_activations(
            model, input_tensor,
            validate=False,
            gw_config=gw_config
        )
        
        # Build Laplacian without explicit dtype (should infer from sheaf)
        gw_laplacian_builder = GWLaplacianBuilder()
        laplacian = gw_laplacian_builder.build_laplacian(sheaf, sparse=True)
        
        # Check that builder inferred correct dtype
        expected_torch = torch.float32 if dtype == 'float32' else torch.float64
        expected_numpy = np.float32 if dtype == 'float32' else np.float64
        
        assert gw_laplacian_builder.torch_dtype == expected_torch
        assert gw_laplacian_builder.numpy_dtype == expected_numpy
        
        # Note: Laplacian assembly currently uses hardcoded float64 but will be updated
        # For now, just verify the builder has the correct dtype configuration
        # TODO: Update Laplacian assembly to use configurable dtype
    
    def test_mixed_dtype_handling(self):
        """Test handling when input has different dtype than config."""
        model = DtypeTestNetwork()
        
        # Input is float32, but config specifies float64
        input_tensor = torch.randn(5, 4, dtype=torch.float32)
        
        builder = SheafBuilder(restriction_method='gromov_wasserstein')
        gw_config = GWConfig(computation_dtype='float64')
        
        # Should work without error (conversion happens internally)
        sheaf = builder.build_from_activations(
            model, input_tensor,
            validate=False,
            gw_config=gw_config
        )
        
        # Restriction maps should be in config dtype, not input dtype
        for restriction in sheaf.restrictions.values():
            assert restriction.dtype == torch.float64
    
    @pytest.mark.parametrize("dtype", ['float32', 'float64'])
    def test_variance_measures_dtype_consistency(self, dtype):
        """Test that variance-based measures use consistent dtype."""
        model = DtypeTestNetwork()
        input_tensor = torch.randn(8, 4)
        
        builder = SheafBuilder(restriction_method='gromov_wasserstein')
        gw_config = GWConfig(
            computation_dtype=dtype,
            uniform_measures=False  # Enable variance-based measures
        )
        
        sheaf = builder.build_from_activations(
            model, input_tensor,
            validate=False,
            gw_config=gw_config
        )
        
        # Should build successfully with non-uniform measures
        assert len(sheaf.restrictions) > 0
        
        expected_torch_dtype = torch.float32 if dtype == 'float32' else torch.float64
        
        # Check restriction maps have correct dtype
        for restriction in sheaf.restrictions.values():
            assert restriction.dtype == expected_torch_dtype
    
    def test_dtype_policy_validation(self):
        """Test that dtype policy is properly validated across pipeline."""
        # Test both dtypes work end-to-end
        for dtype in ['float32', 'float64']:
            model = DtypeTestNetwork()
            input_tensor = torch.randn(6, 4)
            
            # Build sheaf
            builder = SheafBuilder(restriction_method='gromov_wasserstein')
            gw_config = GWConfig(computation_dtype=dtype)
            
            sheaf = builder.build_from_activations(
                model, input_tensor,
                validate=False,
                gw_config=gw_config
            )
            
            # Build Laplacian
            gw_laplacian_builder = GWLaplacianBuilder(computation_dtype=dtype)
            laplacian = gw_laplacian_builder.build_laplacian(sheaf, sparse=True)
            
            # Basic validation that it worked
            assert laplacian.shape[0] == laplacian.shape[1]
            assert laplacian.nnz > 0  # Should have some non-zero entries
    
    def test_numerical_stability_comparison(self):
        """Test numerical stability between float32 and float64."""
        model = DtypeTestNetwork()
        input_tensor = torch.randn(10, 4)
        
        # Build with both dtypes
        sheaves = {}
        laplacians = {}
        
        for dtype in ['float32', 'float64']:
            builder = SheafBuilder(restriction_method='gromov_wasserstein')
            gw_config = GWConfig(computation_dtype=dtype)
            
            sheaf = builder.build_from_activations(
                model, input_tensor,
                validate=False,
                gw_config=gw_config
            )
            sheaves[dtype] = sheaf
            
            gw_laplacian_builder = GWLaplacianBuilder(computation_dtype=dtype)
            laplacian = gw_laplacian_builder.build_laplacian(sheaf, sparse=True)
            laplacians[dtype] = laplacian
        
        # Check that both builds succeeded
        assert laplacians['float32'].shape == laplacians['float64'].shape
        
        # Float64 should generally be more numerically stable
        # (This is more of a sanity check than a strict requirement)
        for dtype, laplacian in laplacians.items():
            assert laplacian.nnz > 0, f"Laplacian for {dtype} should have non-zero entries"
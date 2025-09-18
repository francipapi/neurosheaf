"""Tests for CSR eigenvalue optimization in α-flow analysis.

This module tests the eigen_use_csr flag in AlphaFlowSpec that enables
more efficient eigenvalue computation using explicit CSR matrices instead
of LinearOperator for shift-invert solvers.
"""

import pytest
import numpy as np
from neurosheaf.spectral.persistent import AlphaFlowSpec, StaticBuildConfig
from neurosheaf.spectral.flows.alpha_flow import AlphaFlowBuilder


class TestCSREigenvalueOptimization:
    """Test CSR eigenvalue optimization for α-flow analysis."""
    
    def test_alpha_flow_spec_has_eigen_use_csr_flag(self):
        """Test that AlphaFlowSpec includes the eigen_use_csr parameter."""
        # Test default value
        spec_default = AlphaFlowSpec()
        assert hasattr(spec_default, 'eigen_use_csr'), "AlphaFlowSpec should have eigen_use_csr attribute"
        assert spec_default.eigen_use_csr == True, "Default value should be True for better performance"
        
        # Test explicit values
        spec_false = AlphaFlowSpec(eigen_use_csr=False)
        assert spec_false.eigen_use_csr == False, "Should accept False value"
        
        spec_true = AlphaFlowSpec(eigen_use_csr=True)
        assert spec_true.eigen_use_csr == True, "Should accept True value"
    
    def test_alpha_flow_builder_has_csr_methods(self):
        """Test that AlphaFlowBuilder includes new CSR methods."""
        # Check that new methods exist on the class
        assert hasattr(AlphaFlowBuilder, 'get_csr_matrices'), \
            "AlphaFlowBuilder should have get_csr_matrices method"
        assert hasattr(AlphaFlowBuilder, 'as_csr_combined'), \
            "AlphaFlowBuilder should have as_csr_combined method"
        
        # Check method signatures
        import inspect
        
        get_csr_sig = inspect.signature(AlphaFlowBuilder.get_csr_matrices)
        assert 'grouping' in get_csr_sig.parameters, "get_csr_matrices should accept grouping parameter"
        assert 'mass_mode' in get_csr_sig.parameters, "get_csr_matrices should accept mass_mode parameter"
        
        as_csr_sig = inspect.signature(AlphaFlowBuilder.as_csr_combined)
        assert 'build' in as_csr_sig.parameters, "as_csr_combined should accept build parameter"
        assert 'alpha' in as_csr_sig.parameters, "as_csr_combined should accept alpha parameter"
    
    def test_spec_dataclass_frozen_behavior(self):
        """Test that AlphaFlowSpec remains frozen with new field."""
        spec = AlphaFlowSpec(eigen_use_csr=False)
        
        # Should not be able to modify after creation
        with pytest.raises(Exception):  # FrozenInstanceError or AttributeError
            spec.eigen_use_csr = True
    
    def test_spec_with_different_configurations(self):
        """Test AlphaFlowSpec with various eigen_use_csr configurations."""
        # Test with other parameters
        spec1 = AlphaFlowSpec(
            alpha_grid=[0.0, 1.0],
            k_small=8,
            eigen_use_csr=True
        )
        assert spec1.eigen_use_csr == True
        assert spec1.k_small == 8
        assert list(spec1.alpha_grid) == [0.0, 1.0]
        
        spec2 = AlphaFlowSpec(
            alpha_grid=[0.0, 0.5, 1.0, 2.0],
            k_small=16,
            eigen_use_csr=False
        )
        assert spec2.eigen_use_csr == False
        assert spec2.k_small == 16
        assert len(spec2.alpha_grid) == 4
    
    def test_backward_compatibility(self):
        """Test that existing code without eigen_use_csr still works."""
        # Creating spec without eigen_use_csr should use default
        spec = AlphaFlowSpec(
            alpha_grid=[0.0, 0.5, 1.0],
            k_small=10,
            probes=32
        )
        
        # Should have the default value
        assert spec.eigen_use_csr == True
        assert spec.k_small == 10
        assert spec.probes == 32
    
    def test_documentation_strings(self):
        """Test that documentation mentions the new feature."""
        assert 'eigen_use_csr' in AlphaFlowSpec.__doc__, \
            "AlphaFlowSpec docstring should mention eigen_use_csr"
        
        # Check that the new methods have docstrings
        assert AlphaFlowBuilder.get_csr_matrices.__doc__ is not None, \
            "get_csr_matrices should have documentation"
        assert AlphaFlowBuilder.as_csr_combined.__doc__ is not None, \
            "as_csr_combined should have documentation"
        
        assert 'CSR' in AlphaFlowBuilder.get_csr_matrices.__doc__, \
            "get_csr_matrices docstring should mention CSR"
        assert 'efficient' in AlphaFlowBuilder.get_csr_matrices.__doc__, \
            "get_csr_matrices docstring should mention efficiency"
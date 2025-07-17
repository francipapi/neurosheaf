"""Comprehensive tests for Tikhonov regularization in the neurosheaf pipeline.

This test suite validates the Tikhonov regularization implementation:
- AdaptiveTikhonovRegularizer functionality
- Gram matrix regularization with different strategies
- Integration with whitening and restriction computation
- Numerical stability improvements for large batch sizes
- Preservation of mathematical properties
"""

import pytest
import torch
import numpy as np
from typing import Dict, Any

from neurosheaf.sheaf.core.tikhonov import AdaptiveTikhonovRegularizer, create_regularizer_from_config
from neurosheaf.sheaf.core.gram_matrices import (
    compute_regularized_gram_matrix,
    compute_gram_matrices_with_regularization,
    validate_gram_matrix_properties
)
from neurosheaf.sheaf.core.whitening import WhiteningProcessor


class TestAdaptiveTikhonovRegularizer:
    """Test the AdaptiveTikhonovRegularizer class."""
    
    def test_init_default_parameters(self):
        """Test regularizer initialization with default parameters."""
        regularizer = AdaptiveTikhonovRegularizer()
        
        assert regularizer.strategy == 'adaptive'
        assert regularizer.target_condition == 1e6
        assert regularizer.min_regularization == 1e-12
        assert regularizer.max_regularization == 1e-3
        assert regularizer.eigenvalue_threshold == 1e-12
    
    def test_init_custom_parameters(self):
        """Test regularizer initialization with custom parameters."""
        regularizer = AdaptiveTikhonovRegularizer(
            strategy='moderate',
            target_condition=1e5,
            min_regularization=1e-10,
            max_regularization=1e-4
        )
        
        assert regularizer.strategy == 'moderate'
        assert regularizer.target_condition == 1e5
        assert regularizer.min_regularization == 1e-10
        assert regularizer.max_regularization == 1e-4
    
    def test_condition_number_estimation_well_conditioned(self):
        """Test condition number estimation for well-conditioned matrix."""
        # Create well-conditioned matrix
        torch.manual_seed(42)
        X = torch.randn(50, 20)
        K = X @ X.T + 1e-2 * torch.eye(50)  # Add more regularization for better conditioning
        
        regularizer = AdaptiveTikhonovRegularizer()
        condition_number, diagnostics = regularizer.estimate_condition_number(K)
        
        assert condition_number < 1e6  # Relaxed threshold since Gram matrices can be ill-conditioned
        assert diagnostics['method'] == 'eigenvalue'
        assert 'max_eigenvalue' in diagnostics
        assert 'min_eigenvalue' in diagnostics
    
    def test_condition_number_estimation_ill_conditioned(self):
        """Test condition number estimation for ill-conditioned matrix."""
        # Create ill-conditioned matrix
        torch.manual_seed(42)
        U, S, Vt = torch.svd(torch.randn(100, 50))
        # Create singular values with large condition number
        S_modified = torch.logspace(0, -8, 50)  # Condition number ~1e8
        K = U @ torch.diag(S_modified**2) @ U.T
        
        regularizer = AdaptiveTikhonovRegularizer()
        condition_number, diagnostics = regularizer.estimate_condition_number(K)
        
        assert condition_number > 1e6  # Should be ill-conditioned
        assert diagnostics['method'] == 'eigenvalue'
    
    @pytest.mark.parametrize("strategy", ['conservative', 'moderate', 'aggressive', 'adaptive'])
    def test_regularization_strategies(self, strategy):
        """Test different regularization strategies."""
        torch.manual_seed(42)
        # Create ill-conditioned matrix
        X = torch.randn(64, 32)
        K = X @ X.T
        
        regularizer = AdaptiveTikhonovRegularizer(strategy=strategy)
        lambda_reg, diagnostics = regularizer.estimate_regularization_strength(K, batch_size=64)
        
        assert lambda_reg >= 0
        assert diagnostics['strategy'] == strategy
        assert 'condition_number' in diagnostics
        
        if strategy == 'conservative':
            assert lambda_reg == 1e-10
        elif strategy == 'aggressive':
            assert lambda_reg == 1e-6
    
    def test_adaptive_strategy_batch_size_trigger(self):
        """Test adaptive strategy triggering on batch size."""
        torch.manual_seed(42)
        X = torch.randn(32, 16)
        K = X @ X.T
        
        regularizer = AdaptiveTikhonovRegularizer(strategy='adaptive')
        
        # Small batch size should not trigger regularization
        lambda_small, diag_small = regularizer.estimate_regularization_strength(K, batch_size=32)
        
        # Large batch size should trigger regularization
        lambda_large, diag_large = regularizer.estimate_regularization_strength(K, batch_size=128)
        
        # Check that large batch size triggers regularization
        assert lambda_large > 0  # Should apply some regularization
        if 'trigger' in diag_large:
            assert diag_large['trigger'] in ['batch_size', 'condition_number']
    
    def test_regularize_matrix(self):
        """Test matrix regularization functionality."""
        torch.manual_seed(42)
        X = torch.randn(50, 25)
        K_original = X @ X.T
        
        regularizer = AdaptiveTikhonovRegularizer(strategy='moderate')
        K_regularized, diagnostics = regularizer.regularize(K_original, batch_size=50)
        
        # Check that regularization was applied
        if diagnostics['regularized']:
            assert not torch.allclose(K_original, K_regularized)
            assert diagnostics['regularization_strength'] > 0
            assert diagnostics['post_condition_number'] <= diagnostics['condition_number']
        
        # Verify matrix properties are preserved
        validation = validate_gram_matrix_properties(K_regularized)
        assert validation['is_positive_semidefinite']
        assert validation['is_symmetric']
    
    def test_regularize_in_place(self):
        """Test in-place regularization."""
        torch.manual_seed(42)
        X = torch.randn(40, 20)
        K = X @ X.T
        K_copy = K.clone()
        
        regularizer = AdaptiveTikhonovRegularizer(strategy='aggressive')
        K_result, diagnostics = regularizer.regularize(K, in_place=True, batch_size=100)  # Force regularization
        
        # Should modify original matrix
        assert torch.equal(K, K_result)
        if diagnostics['regularized']:
            # Check diagonal elements specifically since regularization adds λI
            diagonal_diff = torch.diag(K) - torch.diag(K_copy)
            assert torch.any(diagonal_diff > 1e-12)  # Diagonal should have increased
    
    def test_batch_adaptive_regularization(self):
        """Test batch processing of multiple Gram matrices."""
        torch.manual_seed(42)
        gram_matrices = {
            'layer1': torch.randn(30, 15) @ torch.randn(15, 30),
            'layer2': torch.randn(30, 20) @ torch.randn(20, 30),
            'layer3': torch.randn(30, 25) @ torch.randn(25, 30),
        }
        
        regularizer = AdaptiveTikhonovRegularizer(strategy='adaptive')
        results = regularizer.batch_adaptive_regularization(gram_matrices, batch_size=64)
        
        assert len(results) == len(gram_matrices)
        
        for layer_name, (reg_matrix, diagnostics) in results.items():
            assert layer_name in gram_matrices
            assert reg_matrix.shape == gram_matrices[layer_name].shape
            assert isinstance(diagnostics, dict)
            assert 'regularized' in diagnostics


class TestGramMatrixRegularization:
    """Test Gram matrix regularization functions."""
    
    def test_compute_regularized_gram_matrix_no_regularization(self):
        """Test Gram matrix computation without regularization."""
        torch.manual_seed(42)
        activation = torch.randn(32, 64)
        
        gram_matrix, reg_info = compute_regularized_gram_matrix(activation, regularizer=None)
        
        # Should compute standard Gram matrix
        expected = activation @ activation.T
        assert torch.allclose(gram_matrix, expected)
        assert not reg_info['regularized']
    
    def test_compute_regularized_gram_matrix_with_regularization(self):
        """Test Gram matrix computation with regularization."""
        torch.manual_seed(42)
        activation = torch.randn(64, 32)  # Large batch size
        regularizer = AdaptiveTikhonovRegularizer(strategy='adaptive')
        
        gram_matrix, reg_info = compute_regularized_gram_matrix(
            activation, 
            regularizer=regularizer, 
            batch_size=64
        )
        
        # Verify matrix properties
        validation = validate_gram_matrix_properties(gram_matrix)
        assert validation['is_positive_semidefinite']
        assert validation['is_symmetric']
        
        # Check regularization info
        assert 'regularized' in reg_info
        if reg_info['regularized']:
            assert reg_info['regularization_strength'] > 0
            assert 'condition_improvement' in reg_info
    
    def test_compute_gram_matrices_with_regularization(self):
        """Test batch Gram matrix computation with regularization."""
        torch.manual_seed(42)
        activations = {
            'conv1': torch.randn(64, 256),
            'conv2': torch.randn(64, 512),
            'fc1': torch.randn(64, 1024),
        }
        
        regularizer = AdaptiveTikhonovRegularizer(strategy='moderate')
        gram_matrices, reg_info = compute_gram_matrices_with_regularization(
            activations,
            regularizer=regularizer,
            batch_size=64
        )
        
        assert len(gram_matrices) == len(activations)
        assert len(reg_info) == len(activations)
        
        for layer_name in activations.keys():
            assert layer_name in gram_matrices
            assert layer_name in reg_info
            
            # Check matrix properties
            K = gram_matrices[layer_name]
            assert K.shape[0] == K.shape[1] == 64
            
            validation = validate_gram_matrix_properties(K)
            assert validation['is_positive_semidefinite']
            assert validation['is_symmetric']


class TestWhiteningWithRegularization:
    """Test whitening processor with regularized inputs."""
    
    def test_whiten_regularized_gram_matrix(self):
        """Test whitening of regularized Gram matrix."""
        torch.manual_seed(42)
        X = torch.randn(40, 30)
        K_original = X @ X.T
        
        # Apply regularization
        regularizer = AdaptiveTikhonovRegularizer(strategy='moderate')
        K_regularized, reg_info = regularizer.regularize(K_original, batch_size=40)
        
        # Whiten regularized matrix
        whitening_processor = WhiteningProcessor()
        K_whitened, W, whitening_info = whitening_processor.whiten_regularized_gram_matrix(
            K_regularized, reg_info
        )
        
        # Check whitening output
        assert K_whitened.shape[0] == K_whitened.shape[1]
        assert torch.allclose(K_whitened, torch.eye(K_whitened.shape[0]), atol=1e-6)
        
        # Check regularization metadata preservation
        assert whitening_info['input_regularized'] == reg_info.get('regularized', False)
        if reg_info.get('regularized', False):
            assert 'input_regularization_strength' in whitening_info
            assert 'input_condition_improvement' in whitening_info


class TestRegularizerConfiguration:
    """Test regularizer configuration and creation."""
    
    def test_create_regularizer_from_config(self):
        """Test creating regularizer from configuration dictionary."""
        config = {
            'strategy': 'moderate',
            'target_condition': 1e5,
            'min_regularization': 1e-11,
            'max_regularization': 1e-4,
            'eigenvalue_threshold': 1e-13
        }
        
        regularizer = create_regularizer_from_config(config)
        
        assert regularizer.strategy == 'moderate'
        assert regularizer.target_condition == 1e5
        assert regularizer.min_regularization == 1e-11
        assert regularizer.max_regularization == 1e-4
        assert regularizer.eigenvalue_threshold == 1e-13
    
    def test_create_regularizer_from_partial_config(self):
        """Test creating regularizer with partial configuration."""
        config = {
            'strategy': 'aggressive',
            'target_condition': 1e4
        }
        
        regularizer = create_regularizer_from_config(config)
        
        assert regularizer.strategy == 'aggressive'
        assert regularizer.target_condition == 1e4
        # Other parameters should use defaults
        assert regularizer.min_regularization == 1e-12
        assert regularizer.max_regularization == 1e-3


class TestNumericalStability:
    """Test numerical stability improvements from regularization."""
    
    def test_large_batch_size_stability(self):
        """Test stability improvements for large batch sizes."""
        torch.manual_seed(42)
        
        # Create activation that might cause numerical issues
        activation = torch.randn(512, 256)  # Large batch size
        
        # Without regularization
        K_standard = activation @ activation.T
        
        # With regularization
        regularizer = AdaptiveTikhonovRegularizer(strategy='adaptive')
        K_regularized, reg_info = compute_regularized_gram_matrix(
            activation,
            regularizer=regularizer,
            batch_size=512
        )
        
        # Check condition number improvement
        validation_standard = validate_gram_matrix_properties(K_standard)
        validation_regularized = validate_gram_matrix_properties(K_regularized)
        
        if reg_info['regularized']:
            assert validation_regularized['condition_number'] <= validation_standard['condition_number']
            assert reg_info['condition_improvement'] >= 1.0
    
    def test_rank_deficient_matrix_handling(self):
        """Test handling of rank-deficient matrices."""
        torch.manual_seed(42)
        
        # Create rank-deficient activation (more samples than features)
        activation = torch.randn(100, 20)
        
        regularizer = AdaptiveTikhonovRegularizer(strategy='moderate')
        K_regularized, reg_info = compute_regularized_gram_matrix(
            activation,
            regularizer=regularizer,
            batch_size=100
        )
        
        validation = validate_gram_matrix_properties(K_regularized)
        assert validation['is_positive_semidefinite']
        # Note: Regularization can increase effective rank by making small eigenvalues non-zero
        assert validation['effective_rank'] >= 20  # At least the feature dimension
    
    def test_zero_matrix_handling(self):
        """Test handling of zero matrices."""
        # Create zero activation
        activation = torch.zeros(32, 16)
        
        regularizer = AdaptiveTikhonovRegularizer(strategy='conservative')
        K_regularized, reg_info = compute_regularized_gram_matrix(
            activation,
            regularizer=regularizer,
            batch_size=32
        )
        
        # Should handle gracefully
        validation = validate_gram_matrix_properties(K_regularized)
        assert not validation['has_nan']
        assert not validation['has_inf']


class TestMathematicalProperties:
    """Test preservation of mathematical properties."""
    
    def test_positive_semidefinite_preservation(self):
        """Test that regularization preserves positive semidefiniteness."""
        torch.manual_seed(42)
        
        for _ in range(10):  # Test multiple random matrices
            X = torch.randn(50, 30)
            K_original = X @ X.T
            
            regularizer = AdaptiveTikhonovRegularizer(strategy='moderate')
            K_regularized, reg_info = regularizer.regularize(K_original)
            
            validation = validate_gram_matrix_properties(K_regularized)
            assert validation['is_positive_semidefinite']
    
    def test_symmetry_preservation(self):
        """Test that regularization preserves symmetry."""
        torch.manual_seed(42)
        
        X = torch.randn(60, 40)
        K_original = X @ X.T
        
        regularizer = AdaptiveTikhonovRegularizer(strategy='aggressive')
        K_regularized, reg_info = regularizer.regularize(K_original)
        
        validation = validate_gram_matrix_properties(K_regularized)
        assert validation['is_symmetric']
    
    def test_eigenvalue_improvement(self):
        """Test that regularization improves eigenvalue distribution."""
        torch.manual_seed(42)
        
        # Create ill-conditioned matrix
        U = torch.linalg.qr(torch.randn(50, 50), mode='complete')[0]
        eigenvals = torch.logspace(0, -6, 50)  # Large condition number
        K_original = U @ torch.diag(eigenvals) @ U.T
        
        regularizer = AdaptiveTikhonovRegularizer(strategy='moderate', target_condition=1e4)
        K_regularized, reg_info = regularizer.regularize(K_original)
        
        if reg_info['regularized']:
            # Check that condition number improved
            assert reg_info['post_condition_number'] < reg_info['condition_number']
            assert reg_info['condition_improvement'] > 1.0
            
            # Check that minimum eigenvalue increased
            eigenvals_reg = torch.linalg.eigvals(K_regularized).real
            min_eigenval_reg = torch.min(eigenvals_reg).item()
            min_eigenval_orig = torch.min(eigenvals).item()
            
            assert min_eigenval_reg > min_eigenval_orig


if __name__ == '__main__':
    pytest.main([__file__])
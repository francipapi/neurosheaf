"""Tikhonov regularization for numerical stability in Gram matrices.

This module provides adaptive Tikhonov regularization (ridge regularization) to
improve the numerical conditioning of Gram matrices in the neurosheaf pipeline.
The regularization is applied at the Gram matrix level to preserve the
mathematical properties of the downstream Laplacian computation, particularly
the zero eigenvalues that encode topological information.

Key features:
- Adaptive regularization strength based on condition number
- Batch size-aware parameter selection
- Multiple regularization strategies (conservative, moderate, aggressive)
- Preservation of positive semi-definiteness
- Minimal impact on mathematical interpretation
"""

from typing import Dict, Any, Optional, Tuple, Union
import torch
import numpy as np
import logging

logger = logging.getLogger(__name__)


class AdaptiveTikhonovRegularizer:
    """Adaptive Tikhonov regularization for Gram matrices.
    
    This class implements intelligent regularization that automatically
    adjusts the regularization strength based on matrix conditioning and
    batch size to achieve numerical stability without over-regularization.
    
    Attributes:
        strategy: Regularization strategy ('conservative', 'moderate', 'aggressive', 'adaptive')
        target_condition: Target condition number to achieve
        min_regularization: Minimum regularization strength
        max_regularization: Maximum regularization strength
        eigenvalue_threshold: Threshold for considering eigenvalues as zero
    """
    
    def __init__(self,
                 strategy: str = 'adaptive',
                 target_condition: float = 1e6,
                 min_regularization: float = 1e-12,
                 max_regularization: float = 1e-3,
                 eigenvalue_threshold: float = 1e-12):
        """Initialize the Tikhonov regularizer.
        
        Args:
            strategy: Regularization strategy to use
                - 'conservative': Minimal regularization (λ = 1e-10)
                - 'moderate': Balanced approach (λ based on condition number)
                - 'aggressive': Strong regularization (λ = 1e-6)
                - 'adaptive': Automatically choose based on matrix properties
            target_condition: Target condition number for moderate strategy
            min_regularization: Minimum allowed regularization strength
            max_regularization: Maximum allowed regularization strength
            eigenvalue_threshold: Threshold below which eigenvalues are considered zero
        """
        self.strategy = strategy
        self.target_condition = target_condition
        self.min_regularization = min_regularization
        self.max_regularization = max_regularization
        self.eigenvalue_threshold = eigenvalue_threshold
        
        # Strategy-specific parameters
        self.strategy_params = {
            'conservative': {'lambda': 1e-10},
            'moderate': {'target_condition': target_condition},
            'aggressive': {'lambda': 1e-6},
            'adaptive': {'batch_size_threshold': 64, 'condition_threshold': 1e6}
        }
        
        logger.info(f"Initialized AdaptiveTikhonovRegularizer with strategy: {strategy}")
    
    def estimate_condition_number(self, gram_matrix: torch.Tensor) -> Tuple[float, Dict[str, Any]]:
        """Estimate the condition number of a Gram matrix.
        
        Args:
            gram_matrix: Input Gram matrix
            
        Returns:
            Tuple of (condition_number, diagnostics)
        """
        try:
            # Use eigenvalue decomposition for accurate condition number
            eigenvals = torch.linalg.eigvals(gram_matrix).real
            eigenvals = torch.sort(eigenvals, descending=True)[0]
            
            # Filter out numerical zeros
            pos_eigenvals = eigenvals[eigenvals > self.eigenvalue_threshold]
            
            if len(pos_eigenvals) == 0:
                # Zero matrix
                return float('inf'), {
                    'method': 'eigenvalue',
                    'all_zero': True,
                    'max_eigenvalue': 0.0,
                    'min_eigenvalue': 0.0
                }
            
            max_eig = pos_eigenvals[0].item()
            min_eig = pos_eigenvals[-1].item()
            condition_number = max_eig / min_eig
            
            return condition_number, {
                'method': 'eigenvalue',
                'max_eigenvalue': max_eig,
                'min_eigenvalue': min_eig,
                'num_positive_eigenvalues': len(pos_eigenvals),
                'rank_deficiency': len(eigenvals) - len(pos_eigenvals)
            }
            
        except Exception as e:
            logger.warning(f"Eigenvalue computation failed, using SVD: {e}")
            
            try:
                # Fallback to SVD
                S = torch.linalg.svdvals(gram_matrix)
                S_positive = S[S > self.eigenvalue_threshold]
                
                if len(S_positive) == 0:
                    return float('inf'), {'method': 'svd_failed', 'error': 'all_singular_values_zero'}
                
                condition_number = (S_positive[0] / S_positive[-1]).item()
                
                return condition_number, {
                    'method': 'svd',
                    'max_singular_value': S_positive[0].item(),
                    'min_singular_value': S_positive[-1].item()
                }
                
            except Exception as e2:
                logger.error(f"Both eigenvalue and SVD computation failed: {e2}")
                return float('inf'), {'method': 'failed', 'error': str(e2)}
    
    def estimate_regularization_strength(self, 
                                       gram_matrix: torch.Tensor,
                                       batch_size: Optional[int] = None) -> Tuple[float, Dict[str, Any]]:
        """Estimate optimal regularization strength for a Gram matrix.
        
        Args:
            gram_matrix: Input Gram matrix
            batch_size: Optional batch size for adaptive strategies
            
        Returns:
            Tuple of (regularization_strength, diagnostics)
        """
        condition_number, condition_info = self.estimate_condition_number(gram_matrix)
        
        diagnostics = {
            'condition_number': condition_number,
            'condition_info': condition_info,
            'strategy': self.strategy
        }
        
        if self.strategy == 'conservative':
            lambda_reg = self.strategy_params['conservative']['lambda']
            
        elif self.strategy == 'moderate':
            if condition_number <= self.target_condition:
                lambda_reg = self.min_regularization
            else:
                # Compute regularization to achieve target condition number
                max_eig = condition_info.get('max_eigenvalue', 1.0)
                min_eig = condition_info.get('min_eigenvalue', 1e-12)
                target_min_eig = max_eig / self.target_condition
                lambda_reg = max(target_min_eig - min_eig, self.min_regularization)
            
        elif self.strategy == 'aggressive':
            lambda_reg = self.strategy_params['aggressive']['lambda']
            
        elif self.strategy == 'adaptive':
            # Adaptive strategy based on multiple factors
            params = self.strategy_params['adaptive']
            
            # Factor 1: Batch size
            use_regularization = False
            if batch_size is not None and batch_size >= params['batch_size_threshold']:
                use_regularization = True
                diagnostics['trigger'] = 'batch_size'
            
            # Factor 2: Condition number
            if condition_number > params['condition_threshold']:
                use_regularization = True
                diagnostics['trigger'] = 'condition_number'
            
            if use_regularization:
                # Use moderate approach when regularization is needed
                if condition_number > self.target_condition:
                    max_eig = condition_info.get('max_eigenvalue', 1.0)
                    min_eig = condition_info.get('min_eigenvalue', 1e-12)
                    target_min_eig = max_eig / self.target_condition
                    lambda_reg = max(target_min_eig - min_eig, self.min_regularization)
                else:
                    lambda_reg = self.min_regularization
            else:
                lambda_reg = 0.0  # No regularization needed
                
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
        
        # Clamp to allowed range
        lambda_reg = np.clip(lambda_reg, self.min_regularization, self.max_regularization)
        
        diagnostics['regularization_strength'] = lambda_reg
        diagnostics['regularization_percentage'] = lambda_reg / condition_info.get('max_eigenvalue', 1.0) * 100
        
        return lambda_reg, diagnostics
    
    def regularize(self, 
                   gram_matrix: torch.Tensor,
                   batch_size: Optional[int] = None,
                   in_place: bool = False) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Apply Tikhonov regularization to a Gram matrix.
        
        This adds λI to the Gram matrix: K_regularized = K + λI
        
        Args:
            gram_matrix: Input Gram matrix to regularize
            batch_size: Optional batch size for adaptive strategies
            in_place: Whether to modify the input matrix in place
            
        Returns:
            Tuple of (regularized_matrix, regularization_info)
        """
        # Estimate regularization strength
        lambda_reg, diagnostics = self.estimate_regularization_strength(gram_matrix, batch_size)
        
        # Apply regularization
        if lambda_reg > 0:
            n = gram_matrix.shape[0]
            identity = torch.eye(n, dtype=gram_matrix.dtype, device=gram_matrix.device)
            
            if in_place:
                gram_matrix.add_(identity, alpha=lambda_reg)
                regularized_matrix = gram_matrix
            else:
                regularized_matrix = gram_matrix + lambda_reg * identity
            
            logger.debug(f"Applied Tikhonov regularization with λ={lambda_reg:.2e}")
        else:
            regularized_matrix = gram_matrix if in_place else gram_matrix.clone()
            logger.debug("No regularization applied (λ=0)")
        
        # Compute post-regularization condition number
        if lambda_reg > 0:
            post_condition, post_info = self.estimate_condition_number(regularized_matrix)
            diagnostics['post_condition_number'] = post_condition
            diagnostics['condition_improvement'] = diagnostics['condition_number'] / post_condition
            diagnostics['post_condition_info'] = post_info
        
        diagnostics['regularized'] = lambda_reg > 0
        
        return regularized_matrix, diagnostics
    
    def batch_adaptive_regularization(self, 
                                    gram_matrices: Dict[str, torch.Tensor],
                                    batch_size: int) -> Dict[str, Tuple[torch.Tensor, Dict[str, Any]]]:
        """Apply regularization to a dictionary of Gram matrices.
        
        Args:
            gram_matrices: Dictionary mapping layer names to Gram matrices
            batch_size: Batch size for adaptive parameter selection
            
        Returns:
            Dictionary mapping layer names to (regularized_matrix, diagnostics) tuples
        """
        regularized_matrices = {}
        
        for layer_name, gram_matrix in gram_matrices.items():
            try:
                reg_matrix, diagnostics = self.regularize(gram_matrix, batch_size)
                regularized_matrices[layer_name] = (reg_matrix, diagnostics)
                
                if diagnostics['regularized']:
                    logger.info(f"Layer {layer_name}: Applied λ={diagnostics['regularization_strength']:.2e}, "
                               f"condition number {diagnostics['condition_number']:.2e} → "
                               f"{diagnostics.get('post_condition_number', 'N/A'):.2e}")
                
            except Exception as e:
                logger.error(f"Regularization failed for layer {layer_name}: {e}")
                # Return original matrix if regularization fails
                regularized_matrices[layer_name] = (gram_matrix, {'error': str(e), 'regularized': False})
        
        return regularized_matrices


def create_regularizer_from_config(config: Dict[str, Any]) -> AdaptiveTikhonovRegularizer:
    """Create a Tikhonov regularizer from a configuration dictionary.
    
    Args:
        config: Configuration dictionary with regularization parameters
        
    Returns:
        Configured AdaptiveTikhonovRegularizer instance
    """
    return AdaptiveTikhonovRegularizer(
        strategy=config.get('strategy', 'adaptive'),
        target_condition=config.get('target_condition', 1e6),
        min_regularization=config.get('min_regularization', 1e-12),
        max_regularization=config.get('max_regularization', 1e-3),
        eigenvalue_threshold=config.get('eigenvalue_threshold', 1e-12)
    )
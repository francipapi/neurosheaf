"""Enhanced activation extraction that captures functional operations.

This module provides comprehensive activation extraction that captures both
module-based operations (Conv2d, Linear) and functional operations (ReLU, pooling)
to create complete poset structures for sheaf construction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fx as fx
from typing import Dict, List, Optional, Set, Tuple, Any
import contextlib
from ..utils.logging import setup_logger

logger = setup_logger(__name__)


class FunctionalOperationCapture:
    """Context manager to capture functional operations during forward pass."""
    
    def __init__(self):
        self.activations = {}
        self.operation_counter = 0
        self.original_functions = {}
        self.active = False
        
    def __enter__(self):
        """Start capturing functional operations."""
        self.active = True
        self.operation_counter = 0
        self.activations.clear()
        
        # Store original functions
        self.original_functions = {
            'relu': F.relu,
            'adaptive_avg_pool2d': F.adaptive_avg_pool2d,
            'avg_pool2d': F.avg_pool2d,
            'max_pool2d': F.max_pool2d,
            'flatten': torch.flatten,
            'add': torch.add,
            'cat': torch.cat,
        }
        
        # Create capturing wrappers
        def make_capturing_wrapper(original_func, op_name):
            def wrapper(*args, **kwargs):
                result = original_func(*args, **kwargs)
                if self.active:
                    activation_name = f"{op_name}_{self.operation_counter}"
                    self.operation_counter += 1
                    
                    # Process tensor for consistent shape
                    processed = self._process_tensor(result)
                    if processed is not None:
                        self.activations[activation_name] = processed
                        logger.debug(f"Captured {activation_name}: {processed.shape}")
                
                return result
            return wrapper
        
        # Apply wrappers
        F.relu = make_capturing_wrapper(F.relu, 'relu')
        F.adaptive_avg_pool2d = make_capturing_wrapper(F.adaptive_avg_pool2d, 'adaptive_avg_pool2d')
        F.avg_pool2d = make_capturing_wrapper(F.avg_pool2d, 'avg_pool2d')
        F.max_pool2d = make_capturing_wrapper(F.max_pool2d, 'max_pool2d')
        torch.flatten = make_capturing_wrapper(torch.flatten, 'flatten')
        torch.add = make_capturing_wrapper(torch.add, 'add')
        torch.cat = make_capturing_wrapper(torch.cat, 'cat')
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Restore original functions."""
        self.active = False
        
        # Restore original functions
        F.relu = self.original_functions['relu']
        F.adaptive_avg_pool2d = self.original_functions['adaptive_avg_pool2d']
        F.avg_pool2d = self.original_functions['avg_pool2d']
        F.max_pool2d = self.original_functions['max_pool2d']
        torch.flatten = self.original_functions['flatten']
        torch.add = self.original_functions['add']
        torch.cat = self.original_functions['cat']
    
    def _process_tensor(self, tensor: torch.Tensor) -> Optional[torch.Tensor]:
        """Process tensor to consistent shape for activation storage."""
        if not isinstance(tensor, torch.Tensor):
            return None
            
        if len(tensor.shape) == 4:  # [B, C, H, W] - Conv outputs
            return tensor.mean(dim=[2, 3]).detach()
        elif len(tensor.shape) == 3:  # [B, S, D] - Sequence data
            return tensor.mean(dim=1).detach()
        elif len(tensor.shape) == 2:  # [B, F] - Linear outputs
            return tensor.detach()
        else:
            return tensor.flatten(1).detach()


class EnhancedActivationExtractor:
    """Extract activations from both modules and functional operations."""
    
    def __init__(self, capture_functional: bool = True):
        """Initialize the enhanced activation extractor.
        
        Args:
            capture_functional: Whether to capture functional operations
        """
        self.capture_functional = capture_functional
        
    def extract_comprehensive_activations(self, model: nn.Module, 
                                        input_tensor: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Extract activations from all operations in the model.
        
        Args:
            model: PyTorch model
            input_tensor: Input tensor for forward pass
            
        Returns:
            Dictionary mapping operation names to activation tensors
        """
        logger.info("Starting comprehensive activation extraction")
        
        all_activations = {}
        
        # 1. Extract module activations (standard approach)
        module_activations = self._extract_module_activations(model, input_tensor)
        all_activations.update(module_activations)
        
        # 2. Extract functional activations if enabled
        if self.capture_functional:
            functional_activations = self._extract_functional_activations(model, input_tensor)
            all_activations.update(functional_activations)
        
        logger.info(f"Extracted {len(all_activations)} total activations")
        return all_activations
    
    def _extract_module_activations(self, model: nn.Module, 
                                  input_tensor: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Extract activations from module operations."""
        activations = {}
        
        def create_hook(name: str):
            def hook(module, input, output):
                # Process output tensor
                if len(output.shape) == 4:  # Conv layers
                    processed = output.mean(dim=[2, 3]).detach()
                elif len(output.shape) == 2:  # Linear layers
                    processed = output.detach()
                else:
                    processed = output.flatten(1).detach()
                
                activations[name] = processed
                logger.debug(f"Module {name}: {processed.shape}")
            return hook
        
        # Register hooks on relevant modules
        hooks = []
        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear, nn.ReLU, nn.AdaptiveAvgPool2d, 
                                 nn.AvgPool2d, nn.MaxPool2d, nn.Flatten)):
                hook = module.register_forward_hook(create_hook(name))
                hooks.append(hook)
        
        try:
            with torch.no_grad():
                _ = model(input_tensor)
            
            logger.info(f"Extracted {len(activations)} module activations")
            return activations
            
        finally:
            # Clean up hooks
            for hook in hooks:
                hook.remove()
    
    def _extract_functional_activations(self, model: nn.Module, 
                                      input_tensor: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Extract activations from functional operations."""
        
        with FunctionalOperationCapture() as capture:
            try:
                with torch.no_grad():
                    _ = model(input_tensor)
                
                logger.info(f"Extracted {len(capture.activations)} functional activations")
                return capture.activations
                
            except Exception as e:
                logger.warning(f"Functional activation capture failed: {e}")
                return {}


def create_comprehensive_activation_dict(model: nn.Module, 
                                        input_tensor: torch.Tensor,
                                        capture_functional: bool = True) -> Dict[str, torch.Tensor]:
    """Convenience function to extract all activations from a model.
    
    Args:
        model: PyTorch model
        input_tensor: Input tensor
        capture_functional: Whether to capture functional operations
        
    Returns:
        Dictionary of all extracted activations
    """
    extractor = EnhancedActivationExtractor(capture_functional=capture_functional)
    return extractor.extract_comprehensive_activations(model, input_tensor)


# Integration with existing sheaf construction
def build_sheaf_with_enhanced_activations(model: nn.Module, 
                                        input_tensor: torch.Tensor,
                                        sheaf_builder) -> Any:
    """Build sheaf using enhanced activation extraction.
    
    Args:
        model: PyTorch model
        input_tensor: Input tensor
        sheaf_builder: SheafBuilder instance
        
    Returns:
        Constructed sheaf with comprehensive activations
    """
    # Extract comprehensive activations
    comprehensive_activations = create_comprehensive_activation_dict(
        model, input_tensor, capture_functional=True
    )
    
    # Build sheaf using enhanced activations
    sheaf = sheaf_builder.build_from_activations(
        model, comprehensive_activations, use_gram_matrices=True
    )
    
    # Add metadata about extraction method
    sheaf.metadata['enhanced_extraction'] = True
    sheaf.metadata['functional_operations_captured'] = True
    sheaf.metadata['total_activations_extracted'] = len(comprehensive_activations)
    
    return sheaf

# neurosheaf/sheaf/extraction/activations.py

"""
Enhanced activation extraction based on torch.fx.

This module provides a mathematically sound and robust method for activation
extraction. It traces the model graph using torch.fx and inserts hooks
at the exact nodes of interest. This guarantees that the keys in the
resulting activation dictionary perfectly match the node names in the
FX-generated poset, eliminating the need for fragile name mapping.
"""

import torch
import torch.nn as nn
import torch.fx as fx
from typing import Dict, Any, List, Optional, Callable

# Simple logging setup for this module
import logging
logger = logging.getLogger(__name__)


# Module-level storage for activations
_activation_storage = {}


def _capture_activation(name: str, value: torch.Tensor) -> torch.Tensor:
    """
    Global hook function to capture activations.
    
    Args:
        name: The name of the node whose activation is being captured.
        value: The activation tensor.
        
    Returns:
        The activation tensor (unchanged).
    """
    _activation_storage[name] = value.detach().clone()
    return value


class FXActivationExtractor:
    """
    Extracts activations from a PyTorch model using FX tracing.
    
    This class traces the model using torch.fx and modifies the graph to
    capture intermediate activations. The activations are stored in a
    dictionary keyed by the FX node names.
    """
    def __init__(self):
        pass
    
    def _is_activation_node(self, node: fx.Node) -> bool:
        """Determines if a node is one whose activation we want to track."""
        if node.op in {'placeholder', 'output', 'get_attr'}:
            return False
        
        if node.op == 'call_module':
            return True
        
        if node.op == 'call_function':
            import operator
            # Essential functions for architectures like ResNet and Transformers.
            functional_ops = {
                torch.add, operator.add, operator.iadd,
                torch.cat,
                torch.matmul, torch.bmm,
                nn.functional.relu, nn.functional.gelu, nn.functional.silu,
                nn.functional.layer_norm, nn.functional.batch_norm,
                nn.functional.softmax
            }
            if node.target in functional_ops:
                return True
        return False

    def capture_activations(self, model: nn.Module, *args, 
                          node_filter: Optional[Callable] = None) -> Dict[str, torch.Tensor]:
        """
        Traces the model and executes it to capture activations.

        Args:
            model: The PyTorch model to analyze.
            *args: The input tensors for the model.
            node_filter: Optional function to filter which nodes to capture.
                        Takes a node and returns True if it should be captured.

        Returns:
            A dictionary mapping from FX node name to the activation tensor.
        """
        # Clear global storage
        global _activation_storage
        _activation_storage.clear()
        
        # Trace the model
        traced = fx.symbolic_trace(model)
        
        # Create a new graph with activation capture hooks
        new_graph = fx.Graph()
        env = {}  # Maps old nodes to new nodes
        
        # Copy all nodes and insert hooks where needed
        for node in traced.graph.nodes:
            # Copy the node to the new graph
            new_node = new_graph.node_copy(node, lambda x: env[x])
            env[node] = new_node
            
            # Check if we should capture this node's output
            should_capture = False
            if node_filter:
                should_capture = node_filter(node)
            else:
                # Default: capture outputs from activation-producing nodes
                should_capture = self._is_activation_node(node)
            
            if should_capture and node.op != 'output':
                # Insert a call to capture the activation
                capture_node = new_graph.call_function(
                    _capture_activation,
                    args=(node.name, new_node)
                )
                # Replace references to this node with the capture output
                env[node] = capture_node
        
        # Create a new GraphModule with the modified graph
        new_module = fx.GraphModule(traced, new_graph)
        
        # Run the model with the modified graph
        with torch.no_grad():
            output = new_module(*args)
        
        # Return a copy of the activations
        return _activation_storage.copy()


def extract_activations_fx(model: nn.Module, input_tensor: torch.Tensor) -> Dict[str, torch.Tensor]:
    """
    Convenience function to extract activations using the FXActivationExtractor.

    Args:
        model: The PyTorch model.
        input_tensor: An example input tensor to run the forward pass.

    Returns:
        A dictionary of activations, keyed by their FX graph node names.
    """
    logger.info("Extracting activations using FX-based tracer...")
    extractor = FXActivationExtractor()
    activations = extractor.capture_activations(model, input_tensor)
    logger.info(f"Successfully extracted {len(activations)} activations.")
    logger.debug(f"Activation keys: {list(activations.keys())[:10]}...")  # Show first 10 keys
    return activations
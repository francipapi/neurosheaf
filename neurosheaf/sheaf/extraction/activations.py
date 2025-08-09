
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


def create_single_output_exclusion_filter(model: nn.Module, exclude_final_single_output: bool = True) -> Optional[Callable]:
    """
    Create a node filter that excludes final layers with single outputs to prevent GW degeneracy.
    
    This function analyzes the model architecture and creates a filter that excludes:
    1. Linear layers with output dimension = 1 that are near the final output
    2. Activation functions applied after single-output layers (like Sigmoid on classification output)
    3. Any other operations that produce single-dimensional outputs at the end of the network
    
    Args:
        model: The PyTorch model to analyze
        exclude_final_single_output: Whether to enable the filtering
        
    Returns:
        A node filter function if filtering is enabled, None otherwise
    """
    if not exclude_final_single_output:
        return None
    
    # Trace the model to analyze its structure
    traced = fx.symbolic_trace(model)
    
    # Find nodes that should be excluded
    excluded_nodes = set()
    nodes_list = list(traced.graph.nodes)
    
    # Work backwards from the output to identify problematic final layers
    output_node = None
    for node in reversed(nodes_list):
        if node.op == 'output':
            output_node = node
            break
    
    if not output_node:
        return None
    
    # Track nodes that produce single-dimensional outputs near the end
    final_region_depth = 3  # Look at the last few layers
    
    for i, node in enumerate(reversed(nodes_list)):
        if i >= final_region_depth:
            break
            
        if node.op == 'output':
            continue
            
        # Check if this is a linear layer with single output
        should_exclude = False
        
        if node.op == 'call_module':
            # Get the actual module
            try:
                module_name = str(node.target).replace('layers.', '').replace('layers_', '')
                module = None
                
                # Try to get the module from the model
                for name, mod in model.named_modules():
                    if name == node.target or name.replace('.', '_') == node.target or str(node.target) in name:
                        module = mod
                        break
                
                # Check if it's a linear layer with single output
                if isinstance(module, nn.Linear) and module.out_features == 1:
                    should_exclude = True
                    logger.info(f"Excluding single-output linear layer: {node.name} (out_features=1)")
                    
                # Check if it's an activation after a linear layer
                elif isinstance(module, (nn.Sigmoid, nn.Tanh)) and i < final_region_depth - 1:
                    # Look at the previous node to see if it was a single-output linear
                    prev_node_idx = len(nodes_list) - 1 - i - 1
                    if prev_node_idx >= 0:
                        prev_node = nodes_list[prev_node_idx]
                        if prev_node.name in excluded_nodes:
                            should_exclude = True
                            logger.info(f"Excluding activation after single-output layer: {node.name}")
                            
            except Exception as e:
                logger.debug(f"Could not analyze module for node {node.name}: {e}")
                
        elif node.op == 'call_function':
            # Check for functional activations that might be applied to single outputs
            if node.target in [torch.sigmoid, torch.tanh, nn.functional.sigmoid, nn.functional.tanh]:
                # Check if input comes from an excluded node
                for arg in node.args:
                    if hasattr(arg, 'name') and arg.name in excluded_nodes:
                        should_exclude = True
                        logger.info(f"Excluding functional activation after single-output: {node.name}")
                        break
        
        if should_exclude:
            excluded_nodes.add(node.name)
    
    if excluded_nodes:
        logger.info(f"Created exclusion filter for {len(excluded_nodes)} nodes: {list(excluded_nodes)}")
        
        def node_filter(node: fx.Node) -> bool:
            """Filter function that excludes problematic final layers."""
            # Use default activation detection but exclude problematic nodes
            if node.name in excluded_nodes:
                return False
                
            # Use the default activation detection for all other nodes
            extractor = FXActivationExtractor()
            return extractor._is_activation_node(node)
        
        return node_filter
    else:
        logger.info("No single-output final layers detected, no exclusion filter needed")
        return None


def extract_activations_fx(model: nn.Module, input_tensor: torch.Tensor, 
                          exclude_final_single_output: bool = False) -> Dict[str, torch.Tensor]:
    """
    Convenience function to extract activations using the FXActivationExtractor.

    Args:
        model: The PyTorch model.
        input_tensor: An example input tensor to run the forward pass.
        exclude_final_single_output: Whether to exclude final single-output layers to prevent degeneracy.

    Returns:
        A dictionary of activations, keyed by their FX graph node names.
    """
    logger.info("Extracting activations using FX-based tracer...")
    
    # Create exclusion filter if requested
    node_filter = create_single_output_exclusion_filter(model, exclude_final_single_output)
    
    extractor = FXActivationExtractor()
    activations = extractor.capture_activations(model, input_tensor, node_filter=node_filter)
    
    logger.info(f"Successfully extracted {len(activations)} activations.")
    
    if exclude_final_single_output and node_filter:
        logger.info("Single-output final layer exclusion was applied")
    
    logger.debug(f"Activation keys: {list(activations.keys())[:10]}...")  # Show first 10 keys
    return activations
"""FX node to module name mapping service.

This module provides functionality to map between FX symbolic tracing node names
and PyTorch model.named_modules() names, resolving the fundamental mismatch that
prevents activation data from being matched with FX-extracted poset structures.
"""

import torch
import torch.nn as nn
import torch.fx as fx
from typing import Dict, Set, Optional, Tuple, List
import networkx as nx
from ..utils.logging import setup_logger

logger = setup_logger(__name__)


class FXToModuleNameMapper:
    """Maps FX node names to PyTorch module names for activation matching.
    
    This class solves the critical integration issue where FX symbolic tracing
    generates node names like '_0', '_1', '_2_conv1' while activation hooks
    use named_modules() names like '0.conv1', '1.relu', '2.conv2'.
    
    The mapping process:
    1. Analyze FX graph to identify call_module nodes
    2. Match FX node.target with actual module names from model.named_modules()
    3. Create bidirectional mapping for activation key translation
    """
    
    def __init__(self):
        """Initialize the name mapper."""
        self.fx_to_module = {}
        self.module_to_fx = {}
        
    def build_mapping(self, model: nn.Module, fx_graph: fx.Graph) -> Dict[str, str]:
        """Build mapping between FX node names and module names.
        
        Args:
            model: PyTorch model
            fx_graph: FX symbolic graph from tracing
            
        Returns:
            Dictionary mapping FX node names to module names
        """
        logger.info("Building FX node to module name mapping")
        
        # Get all named modules from the model
        named_modules = dict(model.named_modules())
        
        # Build mapping for call_module nodes
        fx_to_module = {}
        module_to_fx = {}
        
        for node in fx_graph.nodes:
            if node.op == 'call_module':
                fx_name = node.name
                target_name = str(node.target)
                
                # Try exact match first
                if target_name in named_modules:
                    fx_to_module[fx_name] = target_name
                    module_to_fx[target_name] = fx_name
                    logger.debug(f"Exact match: {fx_name} → {target_name}")
                else:
                    # Try to find best match based on module type and position
                    best_match = self._find_best_module_match(
                        node, target_name, named_modules, model
                    )
                    if best_match:
                        fx_to_module[fx_name] = best_match
                        module_to_fx[best_match] = fx_name
                        logger.debug(f"Inferred match: {fx_name} → {best_match}")
                    else:
                        logger.warning(f"No module match found for FX node {fx_name} (target: {target_name})")
        
        # Store mappings
        self.fx_to_module = fx_to_module
        self.module_to_fx = module_to_fx
        
        logger.info(f"Built mapping for {len(fx_to_module)} FX nodes")
        return fx_to_module
    
    def _find_best_module_match(self, fx_node: fx.Node, target_name: str, 
                               named_modules: Dict[str, nn.Module], 
                               model: nn.Module) -> Optional[str]:
        """Find best matching module name for an FX node.
        
        Args:
            fx_node: FX node to match
            target_name: Target name from FX node
            named_modules: Dict of all named modules
            model: Original model
            
        Returns:
            Best matching module name or None
        """
        # Strategy 1: Try common name variations
        variations = [
            target_name,
            target_name.replace('_', '.'),
            target_name.replace('.', '_'),
        ]
        
        for variation in variations:
            if variation in named_modules:
                return variation
        
        # Strategy 2: Match by module type and approximate position
        try:
            # Get the actual module that this FX node refers to
            target_module = getattr(model, target_name, None)
            if target_module is None:
                # Try nested attribute access
                parts = target_name.split('.')
                current = model
                for part in parts:
                    if hasattr(current, part):
                        current = getattr(current, part)
                    else:
                        current = None
                        break
                target_module = current
            
            if target_module is not None:
                target_type = type(target_module)
                
                # Find modules of the same type
                candidates = []
                for name, module in named_modules.items():
                    if type(module) == target_type and name:  # Exclude empty names
                        candidates.append(name)
                
                # For now, use heuristic matching - this could be improved
                if len(candidates) == 1:
                    return candidates[0]
                elif candidates:
                    # Try to find the best match based on name similarity
                    target_lower = target_name.lower()
                    for candidate in candidates:
                        candidate_lower = candidate.lower()
                        if target_lower in candidate_lower or candidate_lower in target_lower:
                            return candidate
                    # If no similarity match, return first candidate
                    return candidates[0]
        
        except Exception as e:
            logger.debug(f"Error in module matching for {target_name}: {e}")
        
        return None
    
    def translate_activations(self, activations: Dict[str, torch.Tensor], 
                            direction: str = "module_to_fx") -> Dict[str, torch.Tensor]:
        """Translate activation dictionary keys between naming schemes.
        
        Args:
            activations: Dictionary with activation tensors
            direction: "module_to_fx" or "fx_to_module"
            
        Returns:
            Dictionary with translated keys
        """
        if direction == "module_to_fx":
            mapping = self.module_to_fx
        elif direction == "fx_to_module":
            mapping = self.fx_to_module
        else:
            raise ValueError(f"Invalid direction: {direction}")
        
        translated = {}
        unmatched_keys = []
        
        for old_key, tensor in activations.items():
            if old_key in mapping:
                new_key = mapping[old_key]
                translated[new_key] = tensor
                logger.debug(f"Translated activation key: {old_key} → {new_key}")
            else:
                # Keep unmatched keys as-is for now
                translated[old_key] = tensor
                unmatched_keys.append(old_key)
        
        if unmatched_keys:
            logger.warning(f"Could not translate {len(unmatched_keys)} activation keys: {unmatched_keys[:5]}...")
        
        logger.info(f"Translated {len(activations) - len(unmatched_keys)}/{len(activations)} activation keys")
        return translated
    
    def get_module_name_for_fx_node(self, fx_node_name: str) -> Optional[str]:
        """Get module name for a specific FX node name.
        
        Args:
            fx_node_name: FX node name
            
        Returns:
            Corresponding module name or None
        """
        return self.fx_to_module.get(fx_node_name)
    
    def get_fx_name_for_module(self, module_name: str) -> Optional[str]:
        """Get FX node name for a specific module name.
        
        Args:
            module_name: Module name from named_modules()
            
        Returns:
            Corresponding FX node name or None
        """
        return self.module_to_fx.get(module_name)
    
    def get_mapping_stats(self) -> Dict[str, int]:
        """Get statistics about the mapping.
        
        Returns:
            Dictionary with mapping statistics
        """
        return {
            'fx_nodes_mapped': len(self.fx_to_module),
            'modules_mapped': len(self.module_to_fx),
            'total_mappings': len(self.fx_to_module)
        }


def create_unified_activation_dict(model: nn.Module, 
                                 activations: Dict[str, torch.Tensor]) -> Tuple[Dict[str, torch.Tensor], FXToModuleNameMapper]:
    """Create activation dict compatible with FX poset extraction.
    
    This is a convenience function that handles the complete workflow:
    1. Trace the model with FX
    2. Build name mapping
    3. Translate activation keys
    
    Args:
        model: PyTorch model
        activations: Activation dict with module names as keys
        
    Returns:
        Tuple of (translated_activations, mapper)
    """
    try:
        # Trace the model
        traced = fx.symbolic_trace(model)
        
        # Build mapping
        mapper = FXToModuleNameMapper()
        mapping = mapper.build_mapping(model, traced.graph)
        
        # Translate activations
        translated_activations = mapper.translate_activations(activations, "module_to_fx")
        
        return translated_activations, mapper
        
    except Exception as e:
        logger.error(f"Failed to create unified activation dict: {e}")
        # Return original activations if mapping fails
        return activations, FXToModuleNameMapper()
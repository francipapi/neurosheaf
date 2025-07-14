"""
Extraction module for FX-based poset and activation extraction.

This module provides utilities for extracting network structure (poset)
and activations using PyTorch's FX framework.
"""

from .fx_poset import FXPosetExtractor
from .activations import FXActivationExtractor, extract_activations_fx
from .name_mapping import FXToModuleNameMapper, create_unified_activation_dict

__all__ = [
    "FXPosetExtractor",
    "FXActivationExtractor", 
    "extract_activations_fx",
    "FXToModuleNameMapper",
    "create_unified_activation_dict",
]
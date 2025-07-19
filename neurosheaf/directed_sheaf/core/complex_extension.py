"""Complex extension of real whitened stalks to complex vector spaces.

This module implements the mathematical extension of real whitened stalks to 
complex vector spaces as required for directed sheaf construction:

Mathematical Foundation:
- Complex Extension: F̃(v) = C^{r_v} = R^{r_v} ⊗_R C
- Preserves whitened structure: Identity inner products maintained
- Zero imaginary part initialization: Real matrices → Complex matrices

The extension maintains exact mathematical properties while enabling
complex-valued restriction maps for directional encoding.

Key Features:
- Preserves whitened coordinate structure
- Maintains identity inner products
- Enables complex restriction map computation
- Provides comprehensive validation
"""

import torch
import numpy as np
from typing import Dict, Any, Optional, Tuple
import networkx as nx

# Import base sheaf structures
from ...sheaf.data_structures import Sheaf

# Simple logging setup
import logging
logger = logging.getLogger(__name__)


class ComplexStalkExtender:
    """Extends real whitened stalks to complex vector spaces.
    
    This class implements the mathematical extension of real whitened stalks
    to complex vector spaces as required for directed sheaf construction.
    The extension follows the mathematical formulation:
    
    F̃(v) = C^{r_v} = R^{r_v} ⊗_R C
    
    Key Properties:
    - Preserves whitened structure (identity inner products)
    - Maintains dimensionality (r_v remains constant)
    - Enables complex-valued restriction maps
    - Provides exact mathematical extension
    
    The extension is performed by adding zero imaginary parts to real matrices,
    ensuring that the mathematical structure is preserved while enabling
    complex arithmetic for directional encoding.
    """
    
    def __init__(self, validate_extension: bool = True, tolerance: float = 1e-12):
        """Initialize the complex stalk extender.
        
        Args:
            validate_extension: Whether to validate mathematical properties
            tolerance: Tolerance for numerical validation
        """
        self.validate_extension = validate_extension
        self.tolerance = tolerance
        logger.debug(f"ComplexStalkExtender initialized with tolerance={tolerance}")
    
    def extend_stalk(self, real_stalk: torch.Tensor) -> torch.Tensor:
        """Extend real whitened stalk to complex vector space.
        
        Mathematical Implementation:
        R^{r_v} → C^{r_v} = R^{r_v} ⊗_R C
        
        The extension adds zero imaginary parts to the real stalk matrices,
        creating complex matrices while preserving the underlying mathematical
        structure. This enables complex-valued restriction maps for directional
        encoding.
        
        Args:
            real_stalk: Real whitened stalk tensor (typically identity matrix)
            
        Returns:
            Complex stalk tensor with zero imaginary part
            
        Raises:
            ValueError: If input is not a real tensor
            RuntimeError: If extension validation fails
        """
        if not isinstance(real_stalk, torch.Tensor):
            raise ValueError("Input must be a torch.Tensor")
        
        if real_stalk.is_complex():
            logger.warning("Input stalk is already complex, returning as-is")
            return real_stalk
        
        # Validate input is real
        if not real_stalk.dtype.is_floating_point:
            raise ValueError(f"Input must be real floating-point tensor, got {real_stalk.dtype}")
        
        # Create complex tensor with zero imaginary part
        complex_stalk = torch.complex(real_stalk, torch.zeros_like(real_stalk))
        
        # Validate extension if requested
        if self.validate_extension:
            self._validate_stalk_extension(real_stalk, complex_stalk)
        
        logger.debug(f"Extended stalk from {real_stalk.shape} real to {complex_stalk.shape} complex")
        return complex_stalk
    
    def extend_sheaf_stalks(self, sheaf: Sheaf) -> Dict[str, torch.Tensor]:
        """Extend all stalks in a sheaf to complex vector spaces.
        
        Processes all stalks in the input sheaf and extends them to complex
        vector spaces while preserving the mathematical structure. This is
        typically used to convert a complete real sheaf to a directed sheaf.
        
        Args:
            sheaf: Real sheaf with whitened stalks
            
        Returns:
            Dictionary mapping node names to complex stalk tensors
            
        Raises:
            ValueError: If sheaf is invalid or contains non-real stalks
        """
        if not isinstance(sheaf, Sheaf):
            raise ValueError("Input must be a Sheaf instance")
        
        if len(sheaf.stalks) == 0:
            logger.warning("Sheaf has no stalks to extend")
            return {}
        
        complex_stalks = {}
        
        for node_name, real_stalk in sheaf.stalks.items():
            try:
                complex_stalk = self.extend_stalk(real_stalk)
                complex_stalks[node_name] = complex_stalk
                logger.debug(f"Extended stalk for node '{node_name}': {real_stalk.shape} → {complex_stalk.shape}")
            except Exception as e:
                logger.error(f"Failed to extend stalk for node '{node_name}': {e}")
                raise RuntimeError(f"Extension failed for node '{node_name}': {e}")
        
        # Validate complete extension
        if self.validate_extension:
            self._validate_sheaf_extension(sheaf.stalks, complex_stalks)
        
        logger.info(f"Extended {len(complex_stalks)} stalks to complex vector spaces")
        return complex_stalks
    
    def extend_with_metadata(self, sheaf: Sheaf) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
        """Extend sheaf stalks with detailed metadata.
        
        Extends all stalks to complex vector spaces and returns detailed
        metadata about the extension process for validation and debugging.
        
        Args:
            sheaf: Real sheaf with whitened stalks
            
        Returns:
            Tuple of (complex_stalks, extension_metadata)
        """
        complex_stalks = self.extend_sheaf_stalks(sheaf)
        
        # Compute extension metadata
        metadata = {
            'extension_method': 'complex_tensor_extension',
            'num_stalks_extended': len(complex_stalks),
            'original_dimensions': {
                name: stalk.shape for name, stalk in sheaf.stalks.items()
            },
            'complex_dimensions': {
                name: stalk.shape for name, stalk in complex_stalks.items()
            },
            'total_real_elements': sum(stalk.numel() for stalk in sheaf.stalks.values()),
            'total_complex_elements': sum(stalk.numel() for stalk in complex_stalks.values()),
            'memory_overhead_factor': 2.0,  # Complex numbers are 2x real numbers
            'validation_passed': self.validate_extension,
            'tolerance': self.tolerance
        }
        
        # Add per-stalk analysis
        stalk_analysis = {}
        for name in complex_stalks.keys():
            real_stalk = sheaf.stalks[name]
            complex_stalk = complex_stalks[name]
            
            stalk_analysis[name] = {
                'real_shape': real_stalk.shape,
                'complex_shape': complex_stalk.shape,
                'real_dtype': real_stalk.dtype,
                'complex_dtype': complex_stalk.dtype,
                'dimension_preserved': real_stalk.shape == complex_stalk.shape,
                'structure_preserved': self._check_structure_preservation(real_stalk, complex_stalk)
            }
        
        metadata['stalk_analysis'] = stalk_analysis
        
        return complex_stalks, metadata
    
    def _validate_stalk_extension(self, real_stalk: torch.Tensor, complex_stalk: torch.Tensor) -> None:
        """Validate that complex extension preserves mathematical structure.
        
        Args:
            real_stalk: Original real stalk
            complex_stalk: Extended complex stalk
            
        Raises:
            RuntimeError: If validation fails
        """
        # Check shapes match
        if real_stalk.shape != complex_stalk.shape:
            raise RuntimeError(f"Shape mismatch after extension: {real_stalk.shape} vs {complex_stalk.shape}")
        
        # Check complex stalk has correct dtype
        if not complex_stalk.is_complex():
            raise RuntimeError("Extended stalk is not complex")
        
        # Check that real part matches original
        real_part = complex_stalk.real
        real_diff = torch.abs(real_part - real_stalk).max()
        if real_diff > self.tolerance:
            raise RuntimeError(f"Real part differs from original: max_diff={real_diff}")
        
        # Check that imaginary part is zero
        imag_part = complex_stalk.imag
        imag_max = torch.abs(imag_part).max()
        if imag_max > self.tolerance:
            raise RuntimeError(f"Imaginary part is not zero: max_imag={imag_max}")
        
        logger.debug(f"Stalk extension validation passed: real_diff={real_diff}, imag_max={imag_max}")
    
    def _validate_sheaf_extension(self, real_stalks: Dict[str, torch.Tensor], 
                                 complex_stalks: Dict[str, torch.Tensor]) -> None:
        """Validate that sheaf extension preserves mathematical structure.
        
        Args:
            real_stalks: Original real stalks
            complex_stalks: Extended complex stalks
            
        Raises:
            RuntimeError: If validation fails
        """
        # Check same number of stalks
        if len(real_stalks) != len(complex_stalks):
            raise RuntimeError(f"Number of stalks mismatch: {len(real_stalks)} vs {len(complex_stalks)}")
        
        # Check same node names
        if set(real_stalks.keys()) != set(complex_stalks.keys()):
            raise RuntimeError("Node names mismatch between real and complex stalks")
        
        # Validate each stalk
        for node_name in real_stalks.keys():
            real_stalk = real_stalks[node_name]
            complex_stalk = complex_stalks[node_name]
            
            try:
                self._validate_stalk_extension(real_stalk, complex_stalk)
            except RuntimeError as e:
                raise RuntimeError(f"Validation failed for node '{node_name}': {e}")
        
        logger.debug("Sheaf extension validation passed for all stalks")
    
    def _check_structure_preservation(self, real_stalk: torch.Tensor, complex_stalk: torch.Tensor) -> Dict[str, Any]:
        """Check if mathematical structure is preserved during extension.
        
        Args:
            real_stalk: Original real stalk
            complex_stalk: Extended complex stalk
            
        Returns:
            Dictionary with structure preservation analysis
        """
        analysis = {
            'shapes_match': real_stalk.shape == complex_stalk.shape,
            'real_part_preserved': True,
            'imaginary_part_zero': True,
            'max_real_diff': 0.0,
            'max_imag_abs': 0.0
        }
        
        try:
            # Check real part preservation
            real_part = complex_stalk.real
            real_diff = torch.abs(real_part - real_stalk).max().item()
            analysis['max_real_diff'] = real_diff
            analysis['real_part_preserved'] = real_diff <= self.tolerance
            
            # Check imaginary part is zero
            imag_part = complex_stalk.imag
            imag_max = torch.abs(imag_part).max().item()
            analysis['max_imag_abs'] = imag_max
            analysis['imaginary_part_zero'] = imag_max <= self.tolerance
            
        except Exception as e:
            logger.warning(f"Structure preservation check failed: {e}")
            analysis['error'] = str(e)
        
        return analysis
    
    def create_complex_identity_stalk(self, dimension: int, dtype: torch.dtype = torch.complex64) -> torch.Tensor:
        """Create a complex identity stalk for testing or initialization.
        
        Args:
            dimension: Dimension of the identity matrix
            dtype: Complex dtype for the tensor
            
        Returns:
            Complex identity tensor
        """
        return torch.eye(dimension, dtype=dtype)
    
    def validate_whitened_property(self, complex_stalk: torch.Tensor) -> Dict[str, Any]:
        """Validate that complex stalk maintains whitened properties.
        
        In whitened coordinates, stalks should be identity matrices (or close to it).
        This method validates that the complex extension maintains this property.
        
        Args:
            complex_stalk: Complex stalk tensor to validate
            
        Returns:
            Dictionary with validation results
        """
        if not complex_stalk.is_complex():
            raise ValueError("Input must be a complex tensor")
        
        # Check if stalk is square
        if len(complex_stalk.shape) != 2 or complex_stalk.shape[0] != complex_stalk.shape[1]:
            return {
                'is_whitened': False,
                'error': 'Stalk is not square matrix',
                'shape': complex_stalk.shape
            }
        
        dim = complex_stalk.shape[0]
        identity = torch.eye(dim, dtype=complex_stalk.dtype, device=complex_stalk.device)
        
        # Compute deviation from identity
        deviation = complex_stalk - identity
        max_deviation = torch.abs(deviation).max().item()
        
        # Check if close to identity
        is_whitened = max_deviation <= self.tolerance
        
        return {
            'is_whitened': is_whitened,
            'max_deviation': max_deviation,
            'tolerance': self.tolerance,
            'dimension': dim,
            'dtype': complex_stalk.dtype,
            'real_part_identity': torch.abs(complex_stalk.real - torch.eye(dim, dtype=torch.float32)).max().item(),
            'imaginary_part_zero': torch.abs(complex_stalk.imag).max().item()
        }
# neurosheaf/spectral/dtype_policy.py
"""
Global dtype policy for spectral operations in neurosheaf.

This module provides a centralized dtype policy system to eliminate dtype drift
between modules and ensure numerical consistency in spectral computations.

The policy enforces float64 for spectral operations by default, which provides
better numerical stability for eigenvalue computations, angle calculations,
and transport matrix operations.
"""

import torch
import numpy as np
from typing import Union, Optional, List, Tuple, Any
from ..sheaf.core.gw_config import GWConfig


class SpectralDtypePolicy:
    """
    Global dtype policy for spectral operations.
    
    🔧 CRITICAL FIX: This class addresses the dtype drift issue where different modules
    use inconsistent dtypes (float32 vs float64), causing needless conversions and
    numerical instabilities in PES tracking.
    
    Key Principles:
    - Single source of truth for spectral computation dtype
    - float64 default for numerical stability in spectral operations
    - Integration with existing GWConfig system when available  
    - One conversion point per module (not scattered throughout code)
    """
    
    # Default policy: float64 for spectral stability
    DEFAULT_SPECTRAL_DTYPE = torch.float64
    DEFAULT_NUMPY_DTYPE = np.float64
    
    def __init__(self, gw_config: Optional[GWConfig] = None):
        """
        Initialize dtype policy, optionally integrating with GWConfig.
        
        Args:
            gw_config: Optional GWConfig to inherit dtype from. If None, uses float64 default.
        """
        if gw_config is not None:
            self._torch_dtype = gw_config.get_torch_dtype()
            self._numpy_dtype = gw_config.get_numpy_dtype()
        else:
            self._torch_dtype = self.DEFAULT_SPECTRAL_DTYPE
            self._numpy_dtype = self.DEFAULT_NUMPY_DTYPE
    
    @property
    def torch_dtype(self) -> torch.dtype:
        """Get the torch dtype for this policy."""
        return self._torch_dtype
    
    @property
    def numpy_dtype(self) -> np.dtype:
        """Get the numpy dtype for this policy."""
        return self._numpy_dtype
    
    def to_policy(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Convert tensor to policy dtype.
        
        This is the primary method for enforcing dtype consistency.
        Use this at module entry points to ensure all computations
        use the same dtype.
        
        Args:
            tensor: Input tensor to convert
            
        Returns:
            Tensor converted to policy dtype
        """
        if tensor.dtype != self.torch_dtype:
            return tensor.to(self.torch_dtype)
        return tensor
    
    def to_policy_multiple(self, *tensors: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """
        Convert multiple tensors to policy dtype.
        
        Convenience method for converting several tensors at once.
        
        Args:
            *tensors: Variable number of tensors to convert
            
        Returns:
            Tuple of tensors converted to policy dtype
        """
        return tuple(self.to_policy(tensor) for tensor in tensors)
    
    def create_zeros(self, *shape: int, device: Optional[torch.device] = None) -> torch.Tensor:
        """Create zeros tensor with policy dtype."""
        return torch.zeros(*shape, dtype=self.torch_dtype, device=device)
    
    def create_ones(self, *shape: int, device: Optional[torch.device] = None) -> torch.Tensor:
        """Create ones tensor with policy dtype."""
        return torch.ones(*shape, dtype=self.torch_dtype, device=device)
    
    def create_eye(self, n: int, m: Optional[int] = None, device: Optional[torch.device] = None) -> torch.Tensor:
        """Create identity tensor with policy dtype."""
        if m is None:
            return torch.eye(n, dtype=self.torch_dtype, device=device)
        else:
            return torch.eye(n, m, dtype=self.torch_dtype, device=device)
    
    def from_numpy(self, array: np.ndarray, device: Optional[torch.device] = None) -> torch.Tensor:
        """Convert numpy array to torch tensor with policy dtype."""
        return torch.from_numpy(array).to(dtype=self.torch_dtype, device=device)
    
    def validate_dtype(self, tensor: torch.Tensor, name: str = "tensor") -> None:
        """
        Validate that tensor follows policy dtype.
        
        Args:
            tensor: Tensor to validate
            name: Name for error messages
            
        Raises:
            ValueError: If tensor dtype doesn't match policy
        """
        if tensor.dtype != self.torch_dtype:
            raise ValueError(
                f"Dtype policy violation: {name} has dtype {tensor.dtype}, "
                f"expected {self.torch_dtype}"
            )
    
    def validate_multiple(self, *tensors_and_names: Tuple[torch.Tensor, str]) -> None:
        """
        Validate multiple tensors follow policy dtype.
        
        Args:
            *tensors_and_names: Tuples of (tensor, name) to validate
            
        Raises:
            ValueError: If any tensor dtype doesn't match policy
        """
        for tensor, name in tensors_and_names:
            self.validate_dtype(tensor, name)
    
    def __str__(self) -> str:
        return f"SpectralDtypePolicy(torch_dtype={self.torch_dtype}, numpy_dtype={self.numpy_dtype})"
    
    def __repr__(self) -> str:
        return self.__str__()


# Global default policy instance for easy access
DEFAULT_SPECTRAL_POLICY = SpectralDtypePolicy()

# Convenience functions using default policy
def to_spectral_dtype(tensor: torch.Tensor) -> torch.Tensor:
    """Convert tensor to default spectral dtype (float64)."""
    return DEFAULT_SPECTRAL_POLICY.to_policy(tensor)

def to_spectral_dtype_multiple(*tensors: torch.Tensor) -> Tuple[torch.Tensor, ...]:
    """Convert multiple tensors to default spectral dtype (float64)."""
    return DEFAULT_SPECTRAL_POLICY.to_policy_multiple(*tensors)

def validate_spectral_dtype(tensor: torch.Tensor, name: str = "tensor") -> None:
    """Validate tensor uses default spectral dtype (float64)."""
    DEFAULT_SPECTRAL_POLICY.validate_dtype(tensor, name)

def validate_spectral_dtypes(*tensors_and_names: Tuple[torch.Tensor, str]) -> None:
    """Validate multiple tensors use default spectral dtype (float64)."""
    DEFAULT_SPECTRAL_POLICY.validate_multiple(*tensors_and_names)

def create_spectral_policy(gw_config: Optional[GWConfig] = None) -> SpectralDtypePolicy:
    """
    Create a SpectralDtypePolicy instance, optionally from GWConfig.
    
    Args:
        gw_config: Optional GWConfig to inherit dtype settings from
        
    Returns:
        SpectralDtypePolicy instance
    """
    return SpectralDtypePolicy(gw_config)

def get_spectral_dtype_from_config(gw_config: GWConfig) -> torch.dtype:
    """
    Get spectral dtype directly from GWConfig.
    
    Convenience function for quick dtype extraction from GWConfig.
    
    Args:
        gw_config: GWConfig instance
        
    Returns:
        Corresponding torch.dtype
    """
    return gw_config.get_torch_dtype()

def validate_config_compatibility(gw_config: GWConfig) -> bool:
    """
    Validate that GWConfig dtype settings are compatible with spectral operations.
    
    Args:
        gw_config: GWConfig to validate
        
    Returns:
        True if compatible, False otherwise
    """
    try:
        # Check if config supports required dtype methods
        torch_dtype = gw_config.get_torch_dtype()
        numpy_dtype = gw_config.get_numpy_dtype()
        
        # Validate supported dtypes
        if torch_dtype not in [torch.float32, torch.float64]:
            return False
        if numpy_dtype not in [np.float32, np.float64]:
            return False
            
        # Check consistency between torch and numpy dtypes
        if torch_dtype == torch.float32 and numpy_dtype != np.float32:
            return False
        if torch_dtype == torch.float64 and numpy_dtype != np.float64:
            return False
            
        return True
        
    except Exception:
        return False
"""Custom exception hierarchy for Neurosheaf.

This module defines a comprehensive exception hierarchy for the Neurosheaf framework.
All exceptions inherit from the base NeurosheafError class, providing consistent
error handling and categorization throughout the codebase.

Exception Categories:
- ValidationError: Input validation failures
- ComputationError: Numerical computation failures
- MemoryError: Memory-related issues
- ArchitectureError: Unsupported model architectures
- ConfigurationError: Configuration and setup issues
- ConvergenceError: Algorithm convergence failures
"""

from typing import Optional, Any, Dict


class NeurosheafError(Exception):
    """Base exception for all Neurosheaf errors.
    
    This is the root exception class that all other Neurosheaf exceptions
    inherit from. It provides common functionality for error handling,
    logging, and debugging.
    
    Attributes:
        message: Error message
        context: Additional context information
        recoverable: Whether the error is potentially recoverable
    """
    
    def __init__(
        self,
        message: str,
        context: Optional[Dict[str, Any]] = None,
        recoverable: bool = False
    ):
        """Initialize base exception.
        
        Args:
            message: Error message
            context: Additional context information
            recoverable: Whether the error is potentially recoverable
        """
        super().__init__(message)
        self.message = message
        self.context = self._validate_context(context or {})
        self.recoverable = recoverable
    
    def __str__(self) -> str:
        """Return string representation of the exception."""
        base_msg = self.message
        if self.context:
            context_str = ", ".join(f"{k}={v}" for k, v in self.context.items())
            base_msg += f" (Context: {context_str})"
        return base_msg
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for serialization."""
        return {
            "type": self.__class__.__name__,
            "message": self.message,
            "context": self.context,
            "recoverable": self.recoverable,
        }
    
    def _validate_context(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and sanitize context dictionary.
        
        Args:
            context: Context dictionary to validate
            
        Returns:
            Validated context dictionary
        """
        if not isinstance(context, dict):
            return {"invalid_context": f"Context must be dict, got {type(context).__name__}"}
        
        # Check for extremely large contexts
        max_context_size = 1000  # Maximum number of context items
        max_value_size = 10000   # Maximum size of string representations
        
        if len(context) > max_context_size:
            return {
                "context_truncated": f"Context too large ({len(context)} items > {max_context_size})",
                "context_sample": dict(list(context.items())[:10])  # Show first 10 items
            }
        
        # Check for circular references
        visited_objects = set()
        
        def check_circular_reference(obj, path=""):
            """Check for circular references in nested objects."""
            obj_id = id(obj)
            if obj_id in visited_objects:
                return True
            
            visited_objects.add(obj_id)
            
            # Check nested structures
            if isinstance(obj, dict):
                for k, v in obj.items():
                    if check_circular_reference(v, f"{path}.{k}"):
                        return True
            elif isinstance(obj, (list, tuple)):
                for i, v in enumerate(obj):
                    if check_circular_reference(v, f"{path}[{i}]"):
                        return True
            
            visited_objects.remove(obj_id)
            return False
        
        validated_context = {}
        for key, value in context.items():
            # Sanitize key
            if not isinstance(key, str):
                key = str(key)
            if len(key) > 100:
                key = key[:97] + "..."
            
            # Check for circular references
            if check_circular_reference(value):
                validated_context[key] = f"<circular reference detected in {type(value).__name__}>"
                continue
            
            # Sanitize value
            try:
                str_value = str(value)
                if len(str_value) > max_value_size:
                    validated_context[key] = str_value[:max_value_size-3] + "..."
                else:
                    validated_context[key] = value
            except Exception:
                # If value can't be converted to string, use type info
                validated_context[key] = f"<{type(value).__name__} object>"
        
        return validated_context


class ValidationError(NeurosheafError):
    """Raised when input validation fails.
    
    This exception is raised when inputs to functions or methods fail
    validation checks, such as incorrect tensor shapes, invalid parameters,
    or incompatible data types.
    
    Examples:
        - Tensor shape mismatch
        - Invalid parameter values
        - Incompatible data types
        - Missing required arguments
    """
    
    def __init__(
        self,
        message: str,
        parameter: Optional[str] = None,
        expected: Optional[Any] = None,
        actual: Optional[Any] = None,
        **kwargs
    ):
        """Initialize validation error.
        
        Args:
            message: Error message
            parameter: Name of the parameter that failed validation
            expected: Expected value or type
            actual: Actual value received
            **kwargs: Additional context
        """
        context = kwargs.get('context', {})
        if parameter:
            context['parameter'] = parameter
        if expected is not None:
            context['expected'] = expected
        if actual is not None:
            context['actual'] = actual
        
        super().__init__(message, context, recoverable=True)


class ComputationError(NeurosheafError):
    """Raised when numerical computation fails.
    
    This exception is raised when numerical computations encounter errors
    such as numerical instability, singular matrices, or convergence failures.
    
    Examples:
        - Matrix singularity
        - Numerical overflow/underflow
        - NaN or infinity values
        - Eigenvalue computation failures
    """
    
    def __init__(
        self,
        message: str,
        operation: Optional[str] = None,
        values: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        """Initialize computation error.
        
        Args:
            message: Error message
            operation: Name of the operation that failed
            values: Relevant numerical values
            **kwargs: Additional context
        """
        context = kwargs.get('context', {})
        if operation:
            context['operation'] = operation
        if values:
            context.update(values)
        
        super().__init__(message, context, recoverable=False)


class MemoryError(NeurosheafError):
    """Raised when memory limits are exceeded.
    
    This exception is raised when operations exceed available memory limits,
    either CPU or GPU memory. It includes information about memory usage
    and suggestions for mitigation.
    
    Examples:
        - Out of GPU memory
        - CPU memory exhaustion
        - Matrix too large for available memory
        - Memory leak detection
    """
    
    def __init__(
        self,
        message: str,
        memory_used_mb: Optional[float] = None,
        memory_limit_mb: Optional[float] = None,
        memory_type: str = "cpu",
        **kwargs
    ):
        """Initialize memory error.
        
        Args:
            message: Error message
            memory_used_mb: Memory used in MB
            memory_limit_mb: Memory limit in MB
            memory_type: Type of memory ("cpu" or "gpu")
            **kwargs: Additional context
        """
        context = kwargs.get('context', {})
        if memory_used_mb is not None:
            context['memory_used_mb'] = memory_used_mb
        if memory_limit_mb is not None:
            context['memory_limit_mb'] = memory_limit_mb
        context['memory_type'] = memory_type
        
        super().__init__(message, context, recoverable=True)


class ArchitectureError(NeurosheafError):
    """Raised when model architecture is unsupported or incompatible.
    
    This exception is raised when the neural network architecture cannot
    be processed by the current implementation, or when architectural
    constraints are violated.
    
    Examples:
        - Unsupported layer types
        - Dynamic architectures that cannot be traced
        - Incompatible model formats
        - Architecture parsing failures
    """
    
    def __init__(
        self,
        message: str,
        architecture: Optional[str] = None,
        layer_type: Optional[str] = None,
        **kwargs
    ):
        """Initialize architecture error.
        
        Args:
            message: Error message
            architecture: Name of the architecture
            layer_type: Type of layer that caused the error
            **kwargs: Additional context
        """
        context = kwargs.get('context', {})
        if architecture:
            context['architecture'] = architecture
        if layer_type:
            context['layer_type'] = layer_type
        
        super().__init__(message, context, recoverable=False)


class ConfigurationError(NeurosheafError):
    """Raised when configuration or setup issues occur.
    
    This exception is raised when there are problems with configuration
    files, environment setup, or dependency issues.
    
    Examples:
        - Missing configuration files
        - Invalid configuration values
        - Dependency version conflicts
        - Environment setup failures
    """
    
    def __init__(
        self,
        message: str,
        config_key: Optional[str] = None,
        config_value: Optional[Any] = None,
        **kwargs
    ):
        """Initialize configuration error.
        
        Args:
            message: Error message
            config_key: Configuration key that caused the error
            config_value: Configuration value that caused the error
            **kwargs: Additional context
        """
        context = kwargs.get('context', {})
        if config_key:
            context['config_key'] = config_key
        if config_value is not None:
            context['config_value'] = config_value
        
        super().__init__(message, context, recoverable=True)


class ConvergenceError(NeurosheafError):
    """Raised when iterative algorithms fail to converge.
    
    This exception is raised when iterative algorithms (like eigenvalue
    computation or optimization) fail to converge within specified limits.
    
    Examples:
        - Eigenvalue computation fails to converge
        - Optimization algorithms don't converge
        - Adaptive sampling doesn't stabilize
        - Iterative solvers exceed maximum iterations
    """
    
    def __init__(
        self,
        message: str,
        algorithm: Optional[str] = None,
        iterations: Optional[int] = None,
        tolerance: Optional[float] = None,
        **kwargs
    ):
        """Initialize convergence error.
        
        Args:
            message: Error message
            algorithm: Name of the algorithm that failed
            iterations: Number of iterations attempted
            tolerance: Tolerance that was not achieved
            **kwargs: Additional context
        """
        context = kwargs.get('context', {})
        if algorithm:
            context['algorithm'] = algorithm
        if iterations is not None:
            context['iterations'] = iterations
        if tolerance is not None:
            context['tolerance'] = tolerance
        
        super().__init__(message, context, recoverable=True)


# Convenience functions for common error patterns
def validate_tensor_shape(tensor, expected_shape: tuple, name: str = "tensor"):
    """Validate tensor shape and raise ValidationError if invalid.
    
    Args:
        tensor: Tensor to validate
        expected_shape: Expected shape (use None for variable dimensions)
        name: Name of the tensor for error messages
        
    Raises:
        ValidationError: If tensor shape doesn't match expected shape
    """
    if not hasattr(tensor, 'shape'):
        raise ValidationError(
            f"{name} must have a 'shape' attribute",
            parameter=name,
            expected="tensor-like object",
            actual=type(tensor).__name__
        )
    
    actual_shape = tensor.shape
    if len(actual_shape) != len(expected_shape):
        raise ValidationError(
            f"{name} has wrong number of dimensions",
            parameter=name,
            expected=f"{len(expected_shape)} dimensions",
            actual=f"{len(actual_shape)} dimensions"
        )
    
    for i, (actual, expected) in enumerate(zip(actual_shape, expected_shape)):
        if expected is not None and actual != expected:
            raise ValidationError(
                f"{name} dimension {i} has wrong size",
                parameter=name,
                expected=expected,
                actual=actual
            )


def validate_tensor_values(tensor, name: str = "tensor", allow_nan: bool = False, allow_inf: bool = False):
    """Validate tensor values for NaN and infinity.
    
    Args:
        tensor: Tensor to validate
        name: Name of the tensor for error messages
        allow_nan: Whether to allow NaN values
        allow_inf: Whether to allow infinity values
        
    Raises:
        ValidationError: If tensor contains invalid values
    """
    if not hasattr(tensor, 'shape'):
        raise ValidationError(
            f"{name} must have a 'shape' attribute",
            parameter=name,
            expected="tensor-like object",
            actual=type(tensor).__name__
        )
    
    # Check for NaN values
    if not allow_nan:
        try:
            import numpy as np
            if hasattr(tensor, 'numpy'):
                # PyTorch tensor
                tensor_np = tensor.detach().cpu().numpy()
            else:
                # NumPy array or compatible
                tensor_np = np.asarray(tensor)
            
            if np.any(np.isnan(tensor_np)):
                raise ValidationError(
                    f"{name} contains NaN values",
                    parameter=name,
                    expected="finite values",
                    actual="contains NaN"
                )
        except (ImportError, AttributeError):
            # Fall back to basic checking if numpy not available
            try:
                # Try to convert to float and check
                if hasattr(tensor, 'flatten'):
                    flat_tensor = tensor.flatten()
                    for val in flat_tensor:
                        if str(val) == 'nan':
                            raise ValidationError(
                                f"{name} contains NaN values",
                                parameter=name,
                                expected="finite values",
                                actual="contains NaN"
                            )
            except:
                # If all else fails, skip NaN checking
                pass
    
    # Check for infinity values
    if not allow_inf:
        try:
            import numpy as np
            if hasattr(tensor, 'numpy'):
                # PyTorch tensor
                tensor_np = tensor.detach().cpu().numpy()
            else:
                # NumPy array or compatible
                tensor_np = np.asarray(tensor)
            
            if np.any(np.isinf(tensor_np)):
                raise ValidationError(
                    f"{name} contains infinity values",
                    parameter=name,
                    expected="finite values",
                    actual="contains infinity"
                )
        except (ImportError, AttributeError):
            # Fall back to basic checking if numpy not available
            try:
                # Try to convert to float and check
                if hasattr(tensor, 'flatten'):
                    flat_tensor = tensor.flatten()
                    for val in flat_tensor:
                        if str(val) in ['inf', '-inf']:
                            raise ValidationError(
                                f"{name} contains infinity values",
                                parameter=name,
                                expected="finite values",
                                actual="contains infinity"
                            )
            except:
                # If all else fails, skip infinity checking
                pass


def check_memory_limit(memory_mb: float, limit_mb: float, operation: str = "operation"):
    """Check memory usage against limit and raise MemoryError if exceeded.
    
    Args:
        memory_mb: Current memory usage in MB
        limit_mb: Memory limit in MB
        operation: Name of the operation for error messages
        
    Raises:
        MemoryError: If memory usage exceeds limit
    """
    if memory_mb > limit_mb:
        raise MemoryError(
            f"{operation} would exceed memory limit",
            memory_used_mb=memory_mb,
            memory_limit_mb=limit_mb
        )
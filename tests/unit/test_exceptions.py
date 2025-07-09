"""Unit tests for custom exception hierarchy."""

import pytest
from unittest.mock import MagicMock

from neurosheaf.utils.exceptions import (
    NeurosheafError,
    ValidationError,
    ComputationError,
    MemoryError,
    ArchitectureError,
    ConfigurationError,
    ConvergenceError,
    validate_tensor_shape,
    check_memory_limit,
)


class TestNeurosheafError:
    """Test base NeurosheafError class."""
    
    def test_basic_creation(self):
        """Test basic exception creation."""
        error = NeurosheafError("Test error message")
        
        assert str(error) == "Test error message"
        assert error.message == "Test error message"
        assert error.context == {}
        assert error.recoverable is False
    
    def test_creation_with_context(self):
        """Test exception creation with context."""
        context = {"key1": "value1", "key2": 42}
        error = NeurosheafError("Test error", context=context, recoverable=True)
        
        assert error.context == context
        assert error.recoverable is True
    
    def test_string_representation_with_context(self):
        """Test string representation with context."""
        context = {"param": "value", "count": 10}
        error = NeurosheafError("Test error", context=context)
        
        error_str = str(error)
        assert "Test error" in error_str
        assert "param=value" in error_str
        assert "count=10" in error_str
    
    def test_to_dict_method(self):
        """Test to_dict method."""
        context = {"test": "value"}
        error = NeurosheafError("Test message", context=context, recoverable=True)
        
        error_dict = error.to_dict()
        
        assert error_dict["type"] == "NeurosheafError"
        assert error_dict["message"] == "Test message"
        assert error_dict["context"] == context
        assert error_dict["recoverable"] is True
    
    def test_inheritance_from_exception(self):
        """Test that NeurosheafError inherits from Exception."""
        error = NeurosheafError("Test")
        assert isinstance(error, Exception)


class TestValidationError:
    """Test ValidationError class."""
    
    def test_basic_creation(self):
        """Test basic ValidationError creation."""
        error = ValidationError("Validation failed")
        
        assert isinstance(error, NeurosheafError)
        assert str(error) == "Validation failed"
        assert error.recoverable is True
    
    def test_creation_with_parameter_info(self):
        """Test creation with parameter information."""
        error = ValidationError(
            "Invalid parameter",
            parameter="test_param",
            expected="int",
            actual="str"
        )
        
        assert "Invalid parameter" in str(error)
        assert error.context["parameter"] == "test_param"
        assert error.context["expected"] == "int"
        assert error.context["actual"] == "str"
    
    def test_inheritance_hierarchy(self):
        """Test inheritance hierarchy."""
        error = ValidationError("Test")
        
        assert isinstance(error, ValidationError)
        assert isinstance(error, NeurosheafError)
        assert isinstance(error, Exception)


class TestComputationError:
    """Test ComputationError class."""
    
    def test_basic_creation(self):
        """Test basic ComputationError creation."""
        error = ComputationError("Computation failed")
        
        assert isinstance(error, NeurosheafError)
        assert str(error) == "Computation failed"
        assert error.recoverable is False
    
    def test_creation_with_operation_info(self):
        """Test creation with operation information."""
        values = {"matrix_rank": 5, "condition_number": 1e15}
        error = ComputationError(
            "Matrix is singular",
            operation="matrix_inversion",
            values=values
        )
        
        assert "Matrix is singular" in str(error)
        assert error.context["operation"] == "matrix_inversion"
        assert error.context["matrix_rank"] == 5
        assert error.context["condition_number"] == 1e15


class TestMemoryError:
    """Test MemoryError class (custom, not built-in)."""
    
    def test_basic_creation(self):
        """Test basic MemoryError creation."""
        error = MemoryError("Out of memory")
        
        assert isinstance(error, NeurosheafError)
        assert "Out of memory" in str(error)
        assert error.recoverable is True
    
    def test_creation_with_memory_info(self):
        """Test creation with memory information."""
        error = MemoryError(
            "Memory limit exceeded",
            memory_used_mb=2048.0,
            memory_limit_mb=1024.0,
            memory_type="gpu"
        )
        
        assert "Memory limit exceeded" in str(error)
        assert error.context["memory_used_mb"] == 2048.0
        assert error.context["memory_limit_mb"] == 1024.0
        assert error.context["memory_type"] == "gpu"


class TestArchitectureError:
    """Test ArchitectureError class."""
    
    def test_basic_creation(self):
        """Test basic ArchitectureError creation."""
        error = ArchitectureError("Unsupported architecture")
        
        assert isinstance(error, NeurosheafError)
        assert str(error) == "Unsupported architecture"
        assert error.recoverable is False
    
    def test_creation_with_architecture_info(self):
        """Test creation with architecture information."""
        error = ArchitectureError(
            "Unsupported layer type",
            architecture="ResNet50",
            layer_type="CustomLayer"
        )
        
        assert "Unsupported layer type" in str(error)
        assert error.context["architecture"] == "ResNet50"
        assert error.context["layer_type"] == "CustomLayer"


class TestConfigurationError:
    """Test ConfigurationError class."""
    
    def test_basic_creation(self):
        """Test basic ConfigurationError creation."""
        error = ConfigurationError("Configuration error")
        
        assert isinstance(error, NeurosheafError)
        assert str(error) == "Configuration error"
        assert error.recoverable is True
    
    def test_creation_with_config_info(self):
        """Test creation with configuration information."""
        error = ConfigurationError(
            "Invalid config value",
            config_key="batch_size",
            config_value=-1
        )
        
        assert "Invalid config value" in str(error)
        assert error.context["config_key"] == "batch_size"
        assert error.context["config_value"] == -1


class TestConvergenceError:
    """Test ConvergenceError class."""
    
    def test_basic_creation(self):
        """Test basic ConvergenceError creation."""
        error = ConvergenceError("Algorithm failed to converge")
        
        assert isinstance(error, NeurosheafError)
        assert str(error) == "Algorithm failed to converge"
        assert error.recoverable is True
    
    def test_creation_with_convergence_info(self):
        """Test creation with convergence information."""
        error = ConvergenceError(
            "Eigenvalue computation failed",
            algorithm="power_iteration",
            iterations=1000,
            tolerance=1e-6
        )
        
        assert "Eigenvalue computation failed" in str(error)
        assert error.context["algorithm"] == "power_iteration"
        assert error.context["iterations"] == 1000
        assert error.context["tolerance"] == 1e-6


class TestExceptionHierarchy:
    """Test exception hierarchy behavior."""
    
    def test_all_exceptions_inherit_from_base(self):
        """Test that all exceptions inherit from NeurosheafError."""
        exceptions = [
            ValidationError("test"),
            ComputationError("test"),
            MemoryError("test"),
            ArchitectureError("test"),
            ConfigurationError("test"),
            ConvergenceError("test"),
        ]
        
        for exc in exceptions:
            assert isinstance(exc, NeurosheafError)
    
    def test_catching_base_exception(self):
        """Test that base exception can catch all derived exceptions."""
        exceptions = [
            ValidationError("validation"),
            ComputationError("computation"),
            MemoryError("memory"),
            ArchitectureError("architecture"),
            ConfigurationError("configuration"),
            ConvergenceError("convergence"),
        ]
        
        for exc in exceptions:
            with pytest.raises(NeurosheafError):
                raise exc
    
    def test_specific_exception_catching(self):
        """Test catching specific exception types."""
        with pytest.raises(ValidationError):
            raise ValidationError("test")
        
        with pytest.raises(ComputationError):
            raise ComputationError("test")
        
        with pytest.raises(MemoryError):
            raise MemoryError("test")


class TestValidateTensorShape:
    """Test validate_tensor_shape utility function."""
    
    def test_valid_tensor_shape(self):
        """Test validation with valid tensor shape."""
        # Mock tensor with shape attribute
        mock_tensor = MagicMock()
        mock_tensor.shape = (10, 20)
        
        # Should not raise exception
        validate_tensor_shape(mock_tensor, (10, 20), "test_tensor")
    
    def test_invalid_tensor_shape(self):
        """Test validation with invalid tensor shape."""
        mock_tensor = MagicMock()
        mock_tensor.shape = (10, 20)
        
        with pytest.raises(ValidationError) as exc_info:
            validate_tensor_shape(mock_tensor, (10, 30), "test_tensor")
        
        assert "test_tensor dimension 1 has wrong size" in str(exc_info.value)
        assert exc_info.value.context["expected"] == 30
        assert exc_info.value.context["actual"] == 20
    
    def test_wrong_number_of_dimensions(self):
        """Test validation with wrong number of dimensions."""
        mock_tensor = MagicMock()
        mock_tensor.shape = (10, 20, 30)
        
        with pytest.raises(ValidationError) as exc_info:
            validate_tensor_shape(mock_tensor, (10, 20), "test_tensor")
        
        assert "wrong number of dimensions" in str(exc_info.value)
        assert exc_info.value.context["expected"] == "2 dimensions"
        assert exc_info.value.context["actual"] == "3 dimensions"
    
    def test_tensor_without_shape(self):
        """Test validation with object without shape attribute."""
        invalid_tensor = "not_a_tensor"
        
        with pytest.raises(ValidationError) as exc_info:
            validate_tensor_shape(invalid_tensor, (10, 20), "test_tensor")
        
        assert "must have a 'shape' attribute" in str(exc_info.value)
        assert exc_info.value.context["expected"] == "tensor-like object"
        assert exc_info.value.context["actual"] == "str"
    
    def test_variable_dimensions(self):
        """Test validation with variable dimensions (None)."""
        mock_tensor = MagicMock()
        mock_tensor.shape = (10, 20, 30)
        
        # Should not raise exception with None for variable dimensions
        validate_tensor_shape(mock_tensor, (10, None, 30), "test_tensor")
    
    def test_empty_tensor_shape(self):
        """Test validation with empty tensor shape."""
        mock_tensor = MagicMock()
        mock_tensor.shape = ()
        
        # Should not raise exception
        validate_tensor_shape(mock_tensor, (), "scalar_tensor")


class TestCheckMemoryLimit:
    """Test check_memory_limit utility function."""
    
    def test_memory_under_limit(self):
        """Test memory usage under limit."""
        # Should not raise exception
        check_memory_limit(100.0, 200.0, "test_operation")
    
    def test_memory_over_limit(self):
        """Test memory usage over limit."""
        with pytest.raises(MemoryError) as exc_info:
            check_memory_limit(300.0, 200.0, "test_operation")
        
        assert "test_operation would exceed memory limit" in str(exc_info.value)
        assert exc_info.value.context["memory_used_mb"] == 300.0
        assert exc_info.value.context["memory_limit_mb"] == 200.0
    
    def test_memory_exactly_at_limit(self):
        """Test memory usage exactly at limit."""
        # Should not raise exception
        check_memory_limit(200.0, 200.0, "test_operation")
    
    def test_zero_memory_usage(self):
        """Test with zero memory usage."""
        # Should not raise exception
        check_memory_limit(0.0, 100.0, "test_operation")
    
    def test_negative_memory_usage(self):
        """Test with negative memory usage (edge case)."""
        # Should not raise exception
        check_memory_limit(-10.0, 100.0, "test_operation")


class TestExceptionSerialization:
    """Test exception serialization and deserialization."""
    
    def test_to_dict_all_exceptions(self):
        """Test to_dict method for all exception types."""
        exceptions = [
            NeurosheafError("base", context={"key": "value"}),
            ValidationError("validation", parameter="param"),
            ComputationError("computation", operation="op"),
            MemoryError("memory", memory_used_mb=100.0),
            ArchitectureError("architecture", architecture="ResNet"),
            ConfigurationError("config", config_key="key"),
            ConvergenceError("convergence", algorithm="algo"),
        ]
        
        for exc in exceptions:
            exc_dict = exc.to_dict()
            
            assert "type" in exc_dict
            assert "message" in exc_dict
            assert "context" in exc_dict
            assert "recoverable" in exc_dict
            assert isinstance(exc_dict["context"], dict)
    
    def test_json_serialization(self):
        """Test JSON serialization of exceptions."""
        import json
        
        error = ValidationError(
            "Test error",
            parameter="test_param",
            expected="int",
            actual="str"
        )
        
        error_dict = error.to_dict()
        
        # Should be JSON serializable
        json_str = json.dumps(error_dict)
        reconstructed = json.loads(json_str)
        
        assert reconstructed["type"] == "ValidationError"
        assert reconstructed["message"] == "Test error"
        assert reconstructed["context"]["parameter"] == "test_param"


class TestExceptionEdgeCases:
    """Test edge cases and unusual scenarios."""
    
    def test_empty_message(self):
        """Test exception with empty message."""
        error = NeurosheafError("")
        assert str(error) == ""
    
    def test_none_context(self):
        """Test exception with None context."""
        error = NeurosheafError("test", context=None)
        assert error.context == {}
    
    def test_large_context(self):
        """Test exception with large context."""
        large_context = {f"key_{i}": f"value_{i}" for i in range(1000)}
        error = NeurosheafError("test", context=large_context)
        
        assert len(error.context) == 1000
        assert "key_0" in error.context
        assert "key_999" in error.context
    
    def test_nested_exception_context(self):
        """Test exception with nested context."""
        nested_context = {
            "outer": {
                "inner": {
                    "deep": "value"
                }
            }
        }
        
        error = NeurosheafError("test", context=nested_context)
        assert error.context["outer"]["inner"]["deep"] == "value"
    
    def test_unicode_in_message(self):
        """Test exception with unicode in message."""
        unicode_message = "Error: 测试错误 🚨"
        error = NeurosheafError(unicode_message)
        
        assert str(error) == unicode_message
    
    def test_very_long_message(self):
        """Test exception with very long message."""
        long_message = "Error: " + "x" * 10000
        error = NeurosheafError(long_message)
        
        assert str(error) == long_message


@pytest.mark.phase1
class TestExceptionsPhase1Requirements:
    """Test Phase 1 specific requirements for exceptions."""
    
    def test_exception_hierarchy_ready(self):
        """Test that exception hierarchy is ready for other phases."""
        # Should be able to create all types of exceptions
        exceptions = [
            ValidationError("CKA input validation failed"),
            ComputationError("Eigenvalue computation failed"),
            MemoryError("GPU memory exhausted"),
            ArchitectureError("Unsupported model architecture"),
            ConfigurationError("Invalid configuration"),
            ConvergenceError("Algorithm failed to converge"),
        ]
        
        for exc in exceptions:
            assert isinstance(exc, NeurosheafError)
    
    def test_exceptions_work_with_logging(self):
        """Test that exceptions work well with logging."""
        from neurosheaf.utils.logging import setup_logger
        
        logger = setup_logger("exception_test")
        
        try:
            raise ValidationError("Test validation error")
        except ValidationError as e:
            logger.exception("Caught validation error")
            logger.error(f"Error details: {e.to_dict()}")
        
        # Should not raise any exceptions
        assert True
    
    def test_exceptions_support_debugging(self):
        """Test that exceptions provide good debugging information."""
        error = ComputationError(
            "Matrix computation failed",
            operation="eigenvalue_decomposition",
            values={"matrix_size": 1000, "condition_number": 1e15}
        )
        
        error_dict = error.to_dict()
        
        # Should contain all debugging information
        assert "operation" in error_dict["context"]
        assert "matrix_size" in error_dict["context"]
        assert "condition_number" in error_dict["context"]
        assert error_dict["type"] == "ComputationError"
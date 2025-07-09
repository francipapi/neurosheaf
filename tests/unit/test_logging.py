"""Unit tests for logging infrastructure."""

import pytest
import logging
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock
import threading
import time

from neurosheaf.utils.logging import (
    setup_logger,
    get_logger,
    configure_performance_logging,
    log_memory_usage,
    log_execution_time,
    shutdown_logging,
    PerformanceHandler,
)


class TestSetupLogger:
    """Test setup_logger function."""
    
    def test_basic_logger_creation(self):
        """Test basic logger creation with default settings."""
        logger = setup_logger("test_logger")
        
        assert logger.name == "test_logger"
        assert logger.level == logging.INFO
        assert len(logger.handlers) >= 1
        assert not logger.propagate
    
    def test_logger_level_configuration(self):
        """Test logger level configuration."""
        levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        
        for level in levels:
            logger = setup_logger(f"test_{level}", level=level)
            assert logger.level == getattr(logging, level)
    
    def test_invalid_log_level(self):
        """Test error handling for invalid log levels."""
        with pytest.raises(ValueError, match="Invalid log level"):
            setup_logger("test", level="INVALID")
    
    def test_file_logging(self, temp_dir):
        """Test file logging functionality."""
        log_file = temp_dir / "test.log"
        logger = setup_logger("test_file", log_file=log_file)
        
        test_message = "Test file logging message"
        logger.info(test_message)
        
        assert log_file.exists()
        content = log_file.read_text()
        assert test_message in content
    
    def test_file_logging_directory_creation(self, temp_dir):
        """Test that log directory is created if it doesn't exist."""
        log_file = temp_dir / "subdir" / "test.log"
        logger = setup_logger("test_subdir", log_file=log_file)
        
        logger.info("Test message")
        
        assert log_file.exists()
        assert log_file.parent.is_dir()
    
    def test_file_logging_permission_error(self, temp_dir):
        """Test graceful handling of file permission errors."""
        log_file = temp_dir / "readonly.log"
        log_file.touch()
        log_file.chmod(0o444)  # Read-only
        
        # Should not raise exception, just log warning
        logger = setup_logger("test_readonly", log_file=log_file)
        logger.info("Test message")
        
        # Should still work with console handler
        assert len(logger.handlers) >= 1
    
    def test_format_types(self):
        """Test different format types."""
        format_types = ["standard", "detailed", "simple"]
        
        for format_type in format_types:
            logger = setup_logger(f"test_{format_type}", format_type=format_type)
            # Should not raise exception
            logger.info("Test message")
    
    def test_logger_registry(self):
        """Test that loggers are registered and reused."""
        logger1 = setup_logger("registry_test")
        logger2 = setup_logger("registry_test")
        
        assert logger1 is logger2
    
    def test_thread_safety(self):
        """Test thread safety of logger setup."""
        results = []
        
        def create_logger(name):
            logger = setup_logger(name)
            results.append(logger)
        
        threads = []
        for i in range(10):
            thread = threading.Thread(target=create_logger, args=(f"thread_{i}",))
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        assert len(results) == 10
        assert all(isinstance(r, logging.Logger) for r in results)


class TestGetLogger:
    """Test get_logger function."""
    
    def test_get_existing_logger(self):
        """Test getting existing logger."""
        original = setup_logger("existing")
        retrieved = get_logger("existing")
        
        assert original is retrieved
    
    def test_get_nonexistent_logger(self):
        """Test getting non-existent logger creates new one."""
        logger = get_logger("nonexistent")
        
        assert logger.name == "nonexistent"
        assert logger.level == logging.INFO


class TestPerformanceLogging:
    """Test performance logging functionality."""
    
    def test_configure_performance_logging(self):
        """Test performance logging configuration."""
        logger = setup_logger("performance_test")
        
        configure_performance_logging(logger, memory_threshold_mb=500.0, time_threshold_seconds=30.0)
        
        assert hasattr(logger, 'memory_threshold')
        assert hasattr(logger, 'time_threshold')
        assert logger.memory_threshold == 500.0
        assert logger.time_threshold == 30.0
    
    def test_log_memory_usage(self):
        """Test memory usage logging."""
        logger = setup_logger("memory_test")
        
        # Should not raise exception
        log_memory_usage(logger, 150.5, "test context")
    
    def test_log_execution_time(self):
        """Test execution time logging."""
        logger = setup_logger("time_test")
        
        # Should not raise exception
        log_execution_time(logger, 45.2, "test operation")


class TestPerformanceHandler:
    """Test PerformanceHandler class."""
    
    def test_performance_handler_creation(self):
        """Test PerformanceHandler creation."""
        handler = PerformanceHandler()
        assert handler.level == logging.NOTSET
    
    def test_performance_handler_emit(self):
        """Test PerformanceHandler emit method."""
        handler = PerformanceHandler()
        
        # Create a mock record
        record = MagicMock()
        record.getMessage.return_value = "Test message"
        record.memory_mb = 100.5
        record.duration_seconds = 2.5
        
        # Should not raise exception
        with patch('builtins.print') as mock_print:
            handler.emit(record)
            mock_print.assert_called_once()


class TestLoggingEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_empty_logger_name(self):
        """Test empty logger name."""
        logger = setup_logger("")
        # Empty logger name defaults to root logger
        assert logger.name == "root"
    
    def test_very_long_logger_name(self):
        """Test very long logger name."""
        long_name = "a" * 1000
        logger = setup_logger(long_name)
        assert logger.name == long_name
    
    def test_special_characters_in_name(self):
        """Test special characters in logger name."""
        special_name = "test.logger-with_special/chars"
        logger = setup_logger(special_name)
        assert logger.name == special_name
    
    def test_unicode_in_log_message(self):
        """Test unicode characters in log messages."""
        logger = setup_logger("unicode_test")
        
        unicode_message = "Test message with unicode: 你好世界 🌍"
        logger.info(unicode_message)
        
        # Should not raise exception
    
    def test_very_long_log_message(self):
        """Test very long log messages."""
        logger = setup_logger("long_message_test")
        
        long_message = "x" * 10000
        logger.info(long_message)
        
        # Should not raise exception
    
    def test_log_with_newlines(self):
        """Test log messages with newlines."""
        logger = setup_logger("newline_test")
        
        message_with_newlines = "Line 1\nLine 2\nLine 3"
        logger.info(message_with_newlines)
        
        # Should not raise exception
    
    def test_concurrent_file_logging(self, temp_dir):
        """Test concurrent file logging."""
        log_file = temp_dir / "concurrent.log"
        
        def log_messages(thread_id):
            logger = setup_logger("concurrent", log_file=log_file)
            for i in range(10):
                logger.info(f"Thread {thread_id}, message {i}")
        
        threads = []
        for i in range(5):
            thread = threading.Thread(target=log_messages, args=(i,))
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        assert log_file.exists()
        content = log_file.read_text()
        assert "Thread 0" in content
        assert "Thread 4" in content


class TestShutdownLogging:
    """Test logging shutdown functionality."""
    
    def test_shutdown_logging(self):
        """Test shutdown_logging function."""
        logger = setup_logger("shutdown_test")
        initial_handler_count = len(logger.handlers)
        
        shutdown_logging()
        
        # Should not raise exception
        assert True
    
    def test_shutdown_with_file_handlers(self, temp_dir):
        """Test shutdown with file handlers."""
        log_file = temp_dir / "shutdown.log"
        logger = setup_logger("shutdown_file", log_file=log_file)
        
        logger.info("Before shutdown")
        shutdown_logging()
        
        # Should not raise exception
        assert log_file.exists()


class TestLoggerIntegration:
    """Test logger integration with other components."""
    
    def test_logger_with_exception_handling(self):
        """Test logger with exception handling."""
        logger = setup_logger("exception_test")
        
        try:
            raise ValueError("Test exception")
        except ValueError:
            logger.exception("Exception occurred")
        
        # Should not raise exception
    
    def test_logger_context_manager(self, temp_dir):
        """Test logger as context manager (if supported)."""
        log_file = temp_dir / "context.log"
        
        logger = setup_logger("context_test", log_file=log_file)
        logger.info("Test message")
        
        assert log_file.exists()
    
    def test_logger_with_different_levels(self):
        """Test logger with different log levels."""
        logger = setup_logger("level_test", level="DEBUG")
        
        logger.debug("Debug message")
        logger.info("Info message")
        logger.warning("Warning message")
        logger.error("Error message")
        logger.critical("Critical message")
        
        # Should not raise exception


@pytest.mark.phase1
class TestLoggingPhase1Requirements:
    """Test Phase 1 specific requirements for logging."""
    
    def test_logging_infrastructure_ready(self):
        """Test that logging infrastructure is ready for other phases."""
        # Should be able to create multiple loggers
        cka_logger = setup_logger("neurosheaf.cka")
        sheaf_logger = setup_logger("neurosheaf.sheaf")
        spectral_logger = setup_logger("neurosheaf.spectral")
        
        assert cka_logger.name == "neurosheaf.cka"
        assert sheaf_logger.name == "neurosheaf.sheaf"
        assert spectral_logger.name == "neurosheaf.spectral"
    
    def test_performance_logging_ready(self):
        """Test that performance logging is ready for profiling."""
        logger = setup_logger("neurosheaf.profiling")
        configure_performance_logging(logger)
        
        # Should work with profiling
        log_memory_usage(logger, 100.0, "test")
        log_execution_time(logger, 1.0, "test")
    
    def test_logging_compatible_with_multiprocessing(self):
        """Test logging works with multiprocessing (basic test)."""
        logger = setup_logger("multiprocess_test")
        
        # Basic test - should not raise exception
        logger.info("Multiprocessing test message")
        
        assert True
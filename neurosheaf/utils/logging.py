"""Logging infrastructure for Neurosheaf.

This module provides a unified logging system for the entire Neurosheaf framework.
It supports both console and file logging with configurable levels and formats.

Key features:
- Configurable log levels (DEBUG, INFO, WARNING, ERROR, CRITICAL)
- Console and file output support
- Proper formatting for development and production
- Thread-safe logging
- Contextual logging with function names and line numbers
"""

import logging
import sys
from pathlib import Path
from typing import Optional, Dict, Any
import threading

# Global logger registry to prevent duplicate handlers
_loggers: Dict[str, logging.Logger] = {}
_lock = threading.Lock()


def setup_logger(
    name: str,
    level: str = "INFO",
    log_file: Optional[Path] = None,
    format_type: str = "standard"
) -> logging.Logger:
    """Configure logger with console and optional file output.
    
    Args:
        name: Logger name (typically module name)
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional path to log file
        format_type: Format type ('standard', 'detailed', 'simple')
        
    Returns:
        Configured logger instance
        
    Raises:
        ValueError: If level is invalid
        
    Example:
        >>> logger = setup_logger('neurosheaf.cka', level='DEBUG')
        >>> logger.info('Starting CKA computation')
    """
    with _lock:
        # Return existing logger if already configured
        if name in _loggers:
            return _loggers[name]
        
        # Validate log level
        try:
            numeric_level = getattr(logging, level.upper())
        except AttributeError:
            raise ValueError(f"Invalid log level: {level}")
        
        # Create logger
        logger = logging.getLogger(name)
        logger.setLevel(numeric_level)
        
        # Prevent duplicate handlers
        if logger.handlers:
            logger.handlers.clear()
        
        # Get format based on type
        formatter = _get_formatter(format_type)
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        console_handler.setLevel(numeric_level)
        logger.addHandler(console_handler)
        
        # File handler
        if log_file:
            try:
                log_file.parent.mkdir(parents=True, exist_ok=True)
                file_handler = logging.FileHandler(log_file, mode='a')
                file_handler.setFormatter(_get_formatter('detailed'))
                file_handler.setLevel(numeric_level)
                logger.addHandler(file_handler)
            except (OSError, PermissionError) as e:
                # Log to console if file logging fails
                logger.warning(f"Could not create file handler for {log_file}: {e}")
        
        # Prevent propagation to root logger
        logger.propagate = False
        
        # Store in registry
        _loggers[name] = logger
        
        return logger


def _get_formatter(format_type: str) -> logging.Formatter:
    """Get formatter based on type.
    
    Args:
        format_type: Type of format ('standard', 'detailed', 'simple')
        
    Returns:
        Logging formatter
    """
    formats = {
        'standard': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        'detailed': '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
        'simple': '%(levelname)s - %(message)s'
    }
    
    format_string = formats.get(format_type, formats['standard'])
    return logging.Formatter(format_string)


def get_logger(name: str) -> logging.Logger:
    """Get existing logger or create new one with default settings.
    
    Args:
        name: Logger name
        
    Returns:
        Logger instance
    """
    if name in _loggers:
        return _loggers[name]
    return setup_logger(name)


def configure_performance_logging(
    logger: logging.Logger,
    memory_threshold_mb: float = 1000.0,
    time_threshold_seconds: float = 60.0
) -> None:
    """Configure logger for performance monitoring.
    
    Args:
        logger: Logger to configure
        memory_threshold_mb: Memory usage threshold for warnings
        time_threshold_seconds: Time threshold for warnings
    """
    # Add performance-specific attributes
    logger.memory_threshold = memory_threshold_mb
    logger.time_threshold = time_threshold_seconds
    
    # Create performance-specific handler if needed
    if not any(isinstance(h, PerformanceHandler) for h in logger.handlers):
        perf_handler = PerformanceHandler()
        perf_handler.setLevel(logging.WARNING)
        logger.addHandler(perf_handler)


class PerformanceHandler(logging.Handler):
    """Custom handler for performance-related logging."""
    
    def emit(self, record):
        """Emit performance log record."""
        if hasattr(record, 'memory_mb') or hasattr(record, 'duration_seconds'):
            # Format performance message
            msg = record.getMessage()
            if hasattr(record, 'memory_mb'):
                msg += f" [Memory: {record.memory_mb:.1f}MB]"
            if hasattr(record, 'duration_seconds'):
                msg += f" [Duration: {record.duration_seconds:.2f}s]"
            
            # Use console handler for performance logs
            print(f"PERFORMANCE - {msg}", file=sys.stderr)


def log_memory_usage(logger: logging.Logger, memory_mb: float, context: str = "") -> None:
    """Log memory usage with context.
    
    Args:
        logger: Logger to use
        memory_mb: Memory usage in MB
        context: Context string for the measurement
    """
    record = logger.makeRecord(
        logger.name, logging.INFO, "", 0, 
        f"Memory usage: {memory_mb:.1f}MB {context}", 
        (), None
    )
    record.memory_mb = memory_mb
    logger.handle(record)


def log_execution_time(logger: logging.Logger, duration_seconds: float, context: str = "") -> None:
    """Log execution time with context.
    
    Args:
        logger: Logger to use
        duration_seconds: Duration in seconds
        context: Context string for the measurement
    """
    record = logger.makeRecord(
        logger.name, logging.INFO, "", 0,
        f"Execution time: {duration_seconds:.2f}s {context}",
        (), None
    )
    record.duration_seconds = duration_seconds
    logger.handle(record)


def shutdown_logging() -> None:
    """Shutdown all loggers and clean up handlers."""
    with _lock:
        for logger in _loggers.values():
            for handler in logger.handlers:
                handler.close()
            logger.handlers.clear()
        _loggers.clear()
        logging.shutdown()
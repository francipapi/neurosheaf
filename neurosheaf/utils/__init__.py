"""Utilities for the Neurosheaf package.

This module contains common utilities used throughout the Neurosheaf framework:
- Logging infrastructure
- Custom exception hierarchy
- Performance profiling tools
- Memory management utilities
- Validation helpers
"""

from .logging import setup_logger
from .exceptions import (
    NeurosheafError,
    ValidationError,
    ComputationError,
    MemoryError,
    ArchitectureError,
)
from .profiling import profile_memory

__all__ = [
    "setup_logger",
    "profile_memory",
    "NeurosheafError",
    "ValidationError",
    "ComputationError",
    "MemoryError",
    "ArchitectureError",
]